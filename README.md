ResNet-based ECG Classifier – PhysioNet Challenge 2025 Entry
------------------------------------------------------------
Team ECG_UvA
Overview
--------

This repository contains our submission for the George B. Moody PhysioNet Challenge 2025. 
We developed a deep learning model based on a 1D ResNet architecture to classify Chagas 
disease from 12-lead ECG recordings in WFDB format.

The model was trained using binary labels and includes class balancing and data 
augmentation to improve performance on positive cases.

How to Train the Model
----------------------

To train the model on the provided training data, run:

    python train_model.py -d training_data -m model

Where:
- "training_data" is a folder containing WFDB .hea and .dat files with Chagas labels in the header.
- "model" is the output folder where the trained model (final_model.hdf5) will be saved.

How to Run the Model
--------------------

To run inference on new ECG data using a trained model, run:

    python run_model.py -d holdout_data -m model -o holdout_outputs

Where:
- "holdout_data" is a folder containing WFDB .hea and .dat files (labels not required).
- "model" is the folder containing the trained model file.
- "holdout_outputs" is the folder where the output predictions will be saved.

Model Description
-----------------

- Architecture: 1D ResNet with three residual blocks
- Input shape: 1056 time steps, 12 leads
- Output: Binary prediction with sigmoid activation
- Loss: Binary cross-entropy with class weighting
- Training: 150 epochs with early stopping and learning rate scheduling
- Data Augmentation:
    - Gaussian noise
    - Time shifting
    - Amplitude scaling (applied to positive cases only)
- Framework: TensorFlow / Keras

Requirements
------------

Dependencies are listed in "requirements.txt". Install them with:

    pip install -r requirements.txt

Authors
-------

See "AUTHORS.txt" for contributor names.

License
-------

See "LICENSE.txt" for licensing information.

Notes
-----

- Do NOT edit "train_model.py", "run_model.py", or "helper_code.py", as they are managed by the challenge organizers.
- This submission is compatible with the PhysioNet Challenge 2025 evaluation system and Docker-based testing.

Useful Links
------------

Challenge website: https://physionetchallenges.org/2025/
Evaluation code:   https://github.com/physionetchallenges/evaluation-2025
WFDB documentation: https://wfdb.readthedocs.io/

