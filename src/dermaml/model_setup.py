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
import pandas as pd
import numpy as np
import pickle
import math

# External packages
import cv2 as cv
from sklearn.model_selection import train_test_split
import typer
import yaml
from dermaml import data
import pandas as pd
import typer

# === Reading YAML files

# Custom YAML constructor for joining paths
def join_constructor(loader, node):
    '''
    FIXME review construct sequence
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
    
    for key, path in config['paths'].items():
        if (key == 'output_dir') & (not os.path.exists(path)):
            os.mkdir(path)
        elif not os.path.exists(path):
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

    Docstring generated with an LLM 2025 March 13.
    """
    # --- Check arguments
    if num_best <= 0:
        typer.echo(
            "num-best must be strictly positive",
            err=True)
        raise typer.Abort()
    
    metadata_df = pd.read_csv(config.get('paths', {})['metadata_file'])
    features_df = pd.read_csv(config.get('paths', {})['tabular_feature_file'])

    # --- Prepare join key
    metadata_df.loc[:,config['metadata_ref_header']] = (
        metadata_df[config['metadata_ref_header']].apply(
            lambda x:x[:-len(config['metadata_ref_extension'])]
    ))
    # cross reference filenames
    found_in_metadata = features_df[config['tabular_ref_header']].apply(
        lambda x: x in metadata_df[config['metadata_ref_header']].to_list())
    # map ages to filenames
    filename_age_map = dict(
        zip(metadata_df[config['metadata_ref_header']], 
            metadata_df[config['metadata_target']]))
    mapped_ages = features_df[config['tabular_ref_header']].apply(
            lambda x: filename_age_map.get(x))  
    
    # filter for rows found in metadata
    features_df = features_df.loc[found_in_metadata]  
    # map ages to filenames (assign)
    features_df.loc[:, config['metadata_target']] = mapped_ages

    Xy = features_df.copy()
    # -- Remove join keys
    exclude_from_X = ['Unnamed: 0',
            config['metadata_ref_header'],
            config['tabular_ref_header'],
            config['tabular_ref_hand'],
            config['tabular_ref_target']
            ]
    for col in exclude_from_X:
        if col in Xy.columns:
            Xy.drop(columns=[col],inplace=True)

    # assign train/test split indicator column
    train, test = train_test_split(
        Xy, 
        test_size=0.3, 
        random_state=42    
    )
    return train, test


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
        config:dict,
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
    image_dir = config.get('paths',{})['image_dir']
    metadata_file = config.get('paths',{})['metadata_file']
    segmentation_file = config.get('paths',{})['segmentation_file']
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
