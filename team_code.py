#!/usr/bin/env python

import os
import gc
import datetime
import numpy as np
import wfdb
from glob import glob

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import Callback

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

from helper_code import load_signals

# ---------------------------
# Augmentation Functions
# ---------------------------
def add_gaussian_noise(signal, mean=0.0, std=0.01):
    noise = np.random.normal(mean, std, signal.shape)
    return signal + noise

def time_shift(signal, shift_max=50):
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(signal, shift, axis=0)

def amplitude_scale(signal, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(*scale_range)
    return signal * scale

def augment_positive_samples(X, y):
    X_aug, y_aug = [], []
    for x, label in zip(X, y):
        if label == 1:
            X_aug.append(add_gaussian_noise(x))
            X_aug.append(time_shift(x))
            X_aug.append(amplitude_scale(x))
            y_aug.extend([1, 1, 1])
    return np.array(X_aug), np.array(y_aug)

# ---------------------------
# Data Loader
# ---------------------------
def load_dataset_from_wfdb(folder, fixed_len=1056):
    files = sorted(glob(os.path.join(folder, "*.hea")))
    X, Y = [], []

    for hea_path in files:
        rec_id = os.path.splitext(os.path.basename(hea_path))[0]
        dat_path = os.path.join(folder, rec_id + ".dat")

        if not os.path.exists(dat_path):
            continue

        try:
            record = wfdb.rdrecord(os.path.join(folder, rec_id))
            sig = record.p_signal.T
            if sig.shape[0] != 12 or sig.shape[1] < fixed_len:
                continue
            sig = sig[:, :fixed_len]

            label = None
            with open(hea_path) as f:
                for line in f:
                    if "Chagas label:" in line:
                        label = 1 if "True" in line else 0
                        break
            if label is None:
                continue

            X.append(sig.T)
            Y.append(label)

        except:
            continue

    return np.array(X), np.array(Y)

# ---------------------------
# ResNet Model
# ---------------------------
def resnet(training_log_path: str, classes: int, init_lr: float, tensorboard_dir: str, n_feature_maps: int = 64):
    class CustomCallback(Callback):
        def __init__(self):
            self.dat = []
            self.epoch = 1
        def on_epoch_end(self, batch, logs=None):
            np.save(f"epoch_dat_{self.epoch}.npy", self.dat, allow_pickle=True)
            self.epoch += 1
            gc.collect()

    METRICS = [
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]

    input_layer = keras.layers.Input((1056, 12), dtype='float32')

    def residual_block(input_tensor, filters):
        conv_x = keras.layers.Conv1D(filters, 8, padding='same')(input_tensor)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters, 5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters, 3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        shortcut = keras.layers.Conv1D(filters, 1, padding='same')(input_tensor)
        shortcut = keras.layers.BatchNormalization()(shortcut)

        output = keras.layers.Activation('relu')(keras.layers.add([shortcut, conv_z]))
        return output

    block1 = residual_block(input_layer, n_feature_maps)
    block2 = residual_block(block1, n_feature_maps * 2)
    block3 = residual_block(block2, n_feature_maps * 2)

    gap = keras.layers.GlobalAveragePooling1D()(block3)
    output_layer = keras.layers.Dense(1, activation='sigmoid')(gap)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.SGD(learning_rate=init_lr),
        metrics=METRICS,
        weighted_metrics=["accuracy"]
    )

    file_path = f'resnet_model_{str(datetime.datetime.now())}.hdf5'
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=4, min_lr=1e-7),
        keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, min_delta=1e-4),
        CustomCallback(),
        keras.callbacks.CSVLogger(training_log_path, separator=",", append=False),
        keras.callbacks.TensorBoard(f"{tensorboard_dir}/model_{str(datetime.datetime.now())}", histogram_freq=1)
    ]

    return model, callbacks

# ---------------------------
# Training function
# ---------------------------
def train_model(data_folder, model_folder, verbose):
    X_all, y_all = load_dataset_from_wfdb(data_folder)
    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.15, stratify=y_all, random_state=42)

    X_aug, y_aug = augment_positive_samples(X_train, y_train)
    X_train = np.concatenate([X_train, X_aug])
    y_train = np.concatenate([y_train, y_aug])

    class_weights_array = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train.astype(int))
    class_weights = {0: class_weights_array[0], 1: class_weights_array[1]}

    model, callbacks = resnet(
        training_log_path=os.path.join(model_folder, "training.log"),
        classes=1,
        init_lr=0.1,
        tensorboard_dir=model_folder
    )

    model.fit(
        x=X_train,
        y=y_train.astype(np.float32),
        validation_data=(X_val, y_val.astype(np.float32)),
        callbacks=callbacks,
        shuffle=True,
        batch_size=128,
        epochs=150,
        class_weight=class_weights
    )

    os.makedirs(model_folder, exist_ok=True)
    model.save(os.path.join(model_folder, "final_model.hdf5"))

    if verbose:
        print("Model trained and saved to", os.path.join(model_folder, "final_model.hdf5"))

# ---------------------------
# Load function
# ---------------------------
def load_model(model_folder, verbose):
    model_path = os.path.join(model_folder, "final_model.hdf5")
    model = keras.models.load_model(model_path)
    return {"model": model}

# ---------------------------
# Run function
# ---------------------------
def run_model(record, model, verbose):
    signal, fields = load_signals(record)
    if signal.shape[1] < 1056:
        padded = np.zeros((signal.shape[0], 1056))
        padded[:, :signal.shape[1]] = signal
        signal = padded
    else:
        signal = signal[:, :1056]

    signal = signal.T.astype(np.float32)
    signal = np.expand_dims(signal, axis=0)

    model = model["model"]
    y_prob = model.predict(signal)[0][0]
    y_pred = int(y_prob >= 0.6)
    return y_pred, float(y_prob)
