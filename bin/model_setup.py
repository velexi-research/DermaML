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
Script for running AutoML evaluation.
"""

# --- Imports

# Standard library
import os
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

# External packages
import cv2 as cv
import typer
import yaml
from dermaml import data
import pandas as pd
import typer

# === Reading YAML files

# Custom YAML constructor for joining paths
def join_constructor(loader, node):
    '''
    Written by an LLM 2025 March 13
    '''
    seq = loader.construct_sequence(node)
    return ''.join(map(str, seq))

# Read and validate YAML file
def read_config_yaml(config_file):
    '''
    Reads and parses a YAML configuration file, validating the existence of specified paths.
    _______
    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        dict: The parsed YAML configuration.

    Behavior:
        - Registers a custom YAML constructor for path joining (`!join`).
        - Loads the YAML configuration.
        - Checks that all paths specified under the 'paths' section exist.
        - Aborts execution with an error message if the file or any path is missing.
    
    Generated with an LLM 2025 March 13.
    '''
    # --- Check arguments
    if not os.path.exists(config_file):
        typer.echo(f"config_file '{config_file}' not found", err=True)
        raise typer.Abort()
    # parse reader
    yaml.SafeLoader.add_constructor('!join', join_constructor)
    
    # load file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    for key, path in config.get('paths', {}):
        if not os.path.exists(path):
            typer.echo(
                f'Error finding {key}: {path}',
                err=True
            )
            raise typer.Abort()
        
    return config
        

# --- Read parsed extracted tabular features

def tabular_input(
        config: dict,
        num_best: int = 5,
        ) -> pd.DataFrame:
    """
    Prepares tabular datasets for AutoML evaluation by extracting features and corresponding metadata.
    _______
    Args:
        config (dict): Configuration dictionary containing the following keys:
            - 'tabular_feature_file' (str): Path to the CSV file with feature data.
            - 'metadata_file' (str): Path to the CSV file with metadata.
            - 'metadata_ref_header' (str): Column name in the metadata file for matching.
            - 'tabular_ref_header' (str): Column name in the feature file for matching.
            - 'tabular_ref_extension' (str): File extension or suffix to remove from feature identifiers.
            - 'metadata_ref_extension' (str): File extension or suffix to remove from metadata identifiers.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - X: Array of pre-processed image data (normalized square regions).
            - y: Array of corresponding target values (e.g., age).

    Generated with an LLM 2025 March 13.
    """
    # --- Check arguments

    if num_best <= 0:
        typer.echo(
            "num-best must be strictly positive",
            err=True)
        raise typer.Abort()
    
    # --- Prepare datasets

    # image filename specifiers
    features_df = pd.read_csv(config['tabular_feature_file'])
    metadata_df = pd.read_csv(config['metadata_file'])
    header_metadata = config['metadata_ref_header']
    header_features = config['tabular_ref_header']
    extension_features = config['tabular_ref_extension']
    extension_metadata = config['metadata_ref_extension']

    # --- Join datasets
    features_df.loc[:,header_features] = features_df[header_features].apply(lambda x:x[:-(len(extension_features))])
    metadata_df.loc[:,header_features] = features_df[header_metadata].apply(lambda x:x[:len(extension_metadata)])
    
    # Construct DataFrame for model training and testing
    X = metadata_df.join(features_df.set_index(header_features), on=header_features, how='inner').drop(columns=[header_features])

    return X


# ====== Read and extract image data ========

# --- Read Segmentation File
def read_segmentation_file(segmentation_file:str):
    '''
    Read stored segmentations from SAM2 pipeline
    _______
    
    Returns: dictionary[filename] = arr
    '''    
    # Read file
    with open(segmentation_file, 'rb') as file:
        segmentations = pickle.load(file)
    return segmentations


# --- Image Input Standardization

def square_resize(im, mask=None):
    '''
    Fit an image to a (512, 512) square by first padding the image
    _______
    
    Returns: Image of shape(512, 512, 3)
    '''

    if mask is not None:
        im = np.where(mask[..., None], im, 0)

    # calculate padding distance
    h, w, c = im.shape
    side = max(w, h)
    delta_w = side-w
    delta_h = side-h
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    # add 0 border
    square = cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_CONSTANT,value=0)

    # reshape to size (512, 512)
    norm_square = cv.resize(square, (512, 512))

    return norm_square


# === Prepare Image Dataset ====

def prepare_image_datasets(
        config: dict,
        ) -> None:
    """
    Prepares image datasets for AutoML evaluation by extracting features and corresponding metadata.
    _______
    Args:
        config (dict): Configuration dictionary containing the following keys:
            - 'image_dir' (str): Path to the directory containing image data.
            - 'metadata_file' (str): Path to the CSV file with metadata.
            - 'segmentation_file' (str): Path to the segmentation file.
            - 'metadata_ref_header' (str): Metadata column to match with image names.
            - 'metadata_ref_extension' (str): Suffix to append to the image name for metadata lookup.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - X: Array of pre-processed image data (normalized square regions).
            - y: Array of corresponding target values (e.g., age).

    Behavior:
        - Reads segmentations from the specified file.
        - Loads image samples from the local directory.
        - Extracts metadata for each image based on the reference column and extension.
        - Resizes and normalizes images based on segmentation masks.
        - Returns feature matrix `X` and target values `y`.
    
    Generated with an LLM 2025 March 13.
    """
    # --- Load arguments
    image_dir = config['image_dir']
    metadata_file = config['metadata_file']
    segmentation_file = config['segmentation_file']
    header_metadata = config['metadata_ref_header']
    extension_metadata = config['metadata_ref_extension']

    # === Preparations
    # Read SAM2 segmentations from file
    masks = read_segmentation_file(segmentation_file)
    
    # Read features
    samples, filenames = data.read_local_into_dict(image_dir=image_dir)
    
    # Read metadata
    metadata_df = pd.read_csv(metadata_file)


    # === Edit and store images 
    X, y = [], []
    for i in range(len(filenames)):
        fname = filenames[i]
        root = fname.split('.')[0]

        # define X value
        im = samples[fname]
        mask = masks[root]
        norm_square = square_resize(im, mask)

        # define y 
        instance = metadata_df[metadata_df[header_metadata] == root+extension_metadata]

        X += [norm_square]
        y += [instance.Age.values]
        
    return np.array(X), np.array(y)
