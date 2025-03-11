#!/usr/bin/env python
#
#   Copyright 2022 Velexi Corporation
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""
Script for running CNN training and inference.
"""

# --- Imports

# Standard library
from collections.abc import Callable
from pathlib import Path
import numpy as np
import math
import os


# External packages
import model_setup
from tensorflow.keras import layers, models
import typer
from datetime import datetime
import yaml

#FIXME write results to file with date
#FIXME read in filenames from yaml


# --- Define CNN model
def simple_cnn_model():
    '''
    Last updated: 2025 March 9
    Write a generic, simple CNN regression with 3 convolution steps
    _______
    
    Returns: (np.array, np.array, np.array, np.array) dtype=np.float32
    '''
    model = models.Sequential()

    k = 5
    # First Convolutional Layer
    model.add(layers.Conv2D(32, (k, k), activation='relu', input_shape=(512, 512, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    # Second Convolutional Layer
    model.add(layers.Conv2D(64, (k, k), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Third Convolutional Layer
    model.add(layers.Conv2D(64, (k, k), activation='relu'))

    # Flatten the output and add Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='linear'))

    model.compile(
        optimizer='adam',
        loss='mae',
        metrics=['mae']
    )
    return model

# --- Randomly split entire dataset
def random_split_Xy(
        X:np.array,
        y:np.array,
        percent_split=0.7,
    ):
    '''
    Randomly split entire X, y dataset
    _______
    
    Returns: (np.array, np.array, np.array, np.array) dtype=np.float32
    '''
    image_count = len(X)
    # Check arguments
    train_size = math.floor(image_count * percent_split)

    x_train = np.array(X[:train_size])
    x_test = np.array(X[train_size:])

    # round labels to neearest fifth
    y_train = np.around(y[:train_size]/5, decimals=0)*5
    y_test = np.around(y[train_size:]/5, decimals=0)*5

    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    print('==== Dataset Split')
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)

    return x_train, y_train, x_test, y_test


# === Main program

def main(
        image_dir: Path = '/Users/ntin/Documents/DermaML_local/hawkeye-hands-2024-07-29/images_processed/',
        segmentation_file: Path = '/Users/ntin/Models/sam2/notebooks/2025-02-23_Hand_Segmentations-Corrected-3.pkl',
        metadata_file: Path = "metadata.csv",
        output_dir: Path='',
        split_method: Callable = random_split_Xy,
        experiment_name: str = "cnn",
        ) -> None:
    """
    Run CNN training and inference.

    Results are stored ...
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    today = datetime.today('%Y-$M-%D-%H:%M')

    # --- Check inputs and prepare images
    X, y = model_setup.prepare_image_datasets(
        image_dir=image_dir,
        segmentation_file=segmentation_file,
        metadata_file=metadata_file,
    )

    # --- Split dataset
    x_train, y_train, x_test, y_test = split_method(X, y)

    # --- Initalize Model
    model = simple_cnn_model()

    # --- Perform CNN training
    history = model.fit(
        x_train, y_train, 
        epochs=10, batch_size=32,
        validation_data=(x_test, y_test)
    )

    test_loss, test_mae = model.evaluate(x_test, y_test, verbose=2)
    print(f'\nTest MAE: {test_mae}')

    import matplotlib.pyplot as plt
    plt.plot(history.history['mae'], label='MAE')
    plt.plot(history.history['val_mae'], label = 'val_MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend(loc='lower right')
    plt.savefig(f'{today}_{experiment_name}_loss.png')
    plt.show()


# --- Run app

if __name__ == "__main__":
    typer.run(main)
