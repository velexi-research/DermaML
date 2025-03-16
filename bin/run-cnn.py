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
import dermaml.model_setup as model_setup
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import typer
from datetime import datetime

#FIXME write results to file with date

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


# === Main program

def main(
        config_file: Path = typer.Argument(..., help="Path to the YAML configuration file."),
        output_dir: Path = typer.Option(
            '', "-o", "--output", help="Directory to store output files (e.g., model training results, plots)."
        ),
        split_method: Callable = typer.Option(
            model_setup.random_split_Xy, help="Function used to split the dataset (default: random_split_Xy)."
        ),
        experiment_name: str = typer.Option(
            "cnn", help="Name of the experiment for logging and output file naming."
        ),
    ) -> None:
    """
    Runs Convolutional Neural Network (CNN) training and inference on an image dataset.

    This function:
    - Reads a YAML configuration file for dataset and preprocessing details.
    - Prepares image datasets and splits them into training and testing sets.
    - Initializes a simple CNN model.
    - Trains the CNN model using the prepared data.
    - Evaluates the model on the test dataset and outputs the test mean absolute error (MAE).
    - Plots and saves the training and validation MAE over epochs as a PNG file.

    Args:
        config_file (Path): Path to the YAML configuration file containing dataset and preprocessing details.
        output_dir (Path): Directory to save model outputs, plots, and training results.
        split_method (Callable): Function to split the dataset into training and testing sets. Defaults to `random_split_Xy`.
        experiment_name (str): Name of the experiment used for logging and in output file names (e.g., 'cnn').

    Raises:
        FileNotFoundError: If the specified `output_dir` does not exist, it will be created.
    
    Returns:
        None: Results are saved to files.

    Generated with an LLM 2025 March 13.
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    now = datetime.today('%Y-$M-%D %H-%M')

    # set saving location
    output_loc = config.get('paths', {})['output_dir']
    metadata_used = config.get('paths', {})['metadata_file'].split('/')[-1]
    folder_name = '{}_simple_cnn_{}'.format(now, metadata_used)
    output_dir = os.path.join(output_loc, folder_name)
    os.mkdir(output_dir)

    # --- Check inputs and prepare images
    config = model_setup.read_config_yaml(config_file)
    X, y = model_setup.prepare_image_datasets(
        config=config
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
