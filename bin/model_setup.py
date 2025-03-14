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
    Written by ChatGPT 2025 March 13
    '''
    seq = loader.construct_sequence(node)
    return ''.join(map(str, seq))

# --- Read parsed extracted tabular features

def join_metadata_and_tabular_features(
        config_file:str,
        # features_df:pd.DataFrame,
        # metadata_df:pd.DataFrame
        ) -> pd.DataFrame:
    
    yaml.SafeLoader.add_constructor('!join', join_constructor)
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        
    # image filename specifiers
    features_df = pd.read_csv(config['tabular_feature_file'])
    metadata_df = pd.read_csv(config['metadata_file'])
    header_metadata = config['metadata_ref_header']
    header_features = config['tabular_ref_header']
    extension_features = config['tabular_ref_extension']
    extension_metadata = config['metadata_ref_extension']

    # features store filenames as .png
    features_df.loc[:,header_features] = features_df[header_features].apply(lambda x:x[:-(len(extension_features))])
    
    # features store filenames as .jpeg
    metadata_df.loc[:,header_features] = features_df[header_metadata].apply(lambda x:x[:len(extension_metadata)])
    
    # Construct DataFrame for model training and testing
    X = metadata_df.join(features_df.set_index(header_features), on=header_features, how='inner').drop(columns=[header_features])
    return X

def tabular_input(
        config_file: str,
        num_best: int = 5,
        ) -> pd.DataFrame:
    """
    Run AutoML evaluation.

    Results are output two files: 'model-scores.csv'
    """
    # --- Check arguments
    if not os.path.exists(config_file):
        typer.echo(f"config_file '{config_file}' not found", err=True)
        raise typer.Abort()

    # feature_file = config['tabular_feature_file']
    # metadata_file = config['metadata_file']
    # metadata_target = config['metadata_target']

    # if not os.path.exists(feature_file):
    #     typer.echo(f"feature_file '{feature_file}' not found", err=True)
    #     raise typer.Abort()

    # metadata_path = os.path.exists(metadata_file)
    # if not os.path.isfile(metadata_path):
    #     typer.echo(
    #         f"metadata_file '{metadata_file}' not found in data_dir",
    #         err=True)
    #     raise typer.Abort()

    if num_best <= 0:
        typer.echo(
            "num-best must be strictly positive",
            err=True)
        raise typer.Abort()

    # --- Preparations

    # # Read features
    # features_df = pd.read_csv(feature_file)

    # # Read metadata
    # metadata_df = pd.read_csv(metadata_path)

    X = join_metadata_and_tabular_features(
        config_file=config_file
        # features_df=features_df,
        # metadata_df=metadata_df
    )
    
    return X


# ====== Read and extract image data ========

# --- Read Segmentation File
def read_segmentation_file(segmentation_file:str):
    '''
    Read stored segmentations from SAM2 pipeline
    _______
    
    Returns: dictionary[filename] = arr
    '''
    # Check arguments
    if not os.path.exists(segmentation_file):
        typer.echo(f"segmentation_file '{segmentation_file}' not found", err=True)
        raise typer.Abort()
    
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


# --- Main program

def prepare_image_datasets(
        image_dir: Path = '/Users/ntin/Documents/DermaML_local/hawkeye-hands-2024-07-29/images_processed/',
        segmentation_file: Path = '/Users/ntin/Models/sam2/notebooks/2025-02-23_Hand_Segmentations-Corrected-3.pkl',
        metadata_file: Path = "metadata.csv",
        ) -> None:
    """
    Run AutoML evaluation.

    Results are output two files: 'model-scores.csv'
    """
    from dermaml import data
    import pandas as pd
    # --- Check arguments
    metadata_path = os.path.exists(metadata_file)
    if not os.path.isfile(metadata_path):
        typer.echo(
            f"metadata_file '{metadata_file}' not found in data_dir",
            err=True)
        raise typer.Abort()

    # === Preparations
    # Read SAM2 segmentations from file
    masks = read_segmentation_file(segmentation_file)
    
    # Read features
    samples, filenames = data.read_local_into_dict(image_dir=image_dir)
    
    # Read metadata
    metadata_df = pd.read_csv(metadata_path)
    header_metadata = 'hand_image_file'
    extension_metadata = '.jpeg'

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
