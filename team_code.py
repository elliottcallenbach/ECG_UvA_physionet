#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import os
import gc
import datetime
import numpy as np
import wfdb
from glob import glob

import tensorflow as tf
import tensorflow.keras as keras
import keras.backend as K
from tensorflow.keras.callbacks import Callback

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

from helper_code import load_signals

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    # Load all data from data_folder
    X_all, y_all = load_dataset_from_wfdb(data_folder)

    # Train-validation split (e.g., 85/15 split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.15, stratify=y_all, random_state=42
    )

    # Augment positive-class examples in training set
    X_aug, y_aug = augment_positive_samples(X_train, y_train)
    X_train = np.concatenate([X_train, X_aug])
    y_train = np.concatenate([y_train, y_aug])

    # Compute class weights
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=y_train.astype(int)
    )
    class_weights = {0: class_weights_array[0], 1: class_weights_array[1]}

    # Train model
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

    # Save trained model
    os.makedirs(model_folder, exist_ok=True)
    model.save(os.path.join(model_folder, "final_model.hdf5"))

    if verbose:
        print("Model trained and saved to", os.path.join(model_folder, "final_model.hdf5"))


# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    model_path = os.path.join(model_folder, "final_model.hdf5")
    model = keras.models.load_model(model_path)
    return {"model": model}

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    signal, fields = load_signals(record)

    # Pad or truncate signal to 1056 samples
    if signal.shape[1] < 1056:
        padded = np.zeros((signal.shape[0], 1056))
        padded[:, :signal.shape[1]] = signal
        signal = padded
    else:
        signal = signal[:, :1056]

    # Format: (1, 1056, 12)
    signal = signal.T.astype(np.float32)
    signal = np.expand_dims(signal, axis=0)

    # Predict
    model = model["model"]
    y_prob = model.predict(signal)[0][0]
    y_pred = int(y_prob >= 0.6)

    return y_pred, float(y_prob)