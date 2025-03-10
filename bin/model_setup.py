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

# External packages
import pandas as pd
import typer


# --- Main program

def tabular_input(
        feature_file: Path = "texture_features.csv",
        metadata_file: Path = "metadata.csv",
        num_best: int = 5,
        ) -> pd.DataFrame:
    """
    Run AutoML evaluation.

    Results are output two files: 'model-scores.csv'
    """
    # --- Check arguments

    if not os.path.exists(feature_file):
        typer.echo(f"feature_file '{feature_file}' not found", err=True)
        raise typer.Abort()

    metadata_path = os.path.exists(metadata_file)
    if not os.path.isfile(metadata_path):
        typer.echo(
            f"metadata_file '{metadata_file}' not found in data_dir",
            err=True)
        raise typer.Abort()

    if num_best <= 0:
        typer.echo(
            "num-best must be strictly positive",
            err=True)
        raise typer.Abort()

    # --- Preparations

    # Read features
    features_df = pd.read_csv(feature_file)

    # Read metadata
    metadata_df = pd.read_csv(metadata_path)

    def combine_metadata_and_features(
            features_df:pd.DataFrame,
            metadata_df:pd.DataFrame
            ) -> pd.DataFrame:
            
        # image filename specifiers
        header_metadata = 'hand_image_file'
        header_features = 'filename'
        extension_features = '.png'
        extension_metadata = '.jpeg'

        # features store filenames as .png
        features_df.loc[:,header_features] = features_df['filename'].apply(lambda x:x[:-(len(extension_features))])
        
        # features store filenames as .jpeg
        metadata_df.loc[:,header_features] = features_df[header_metadata].apply(lambda x:x[:len(extension_metadata)])
        
        # Construct DataFrame for model training and testing
        X = metadata_df.join(features_df.set_index(header_features), on=header_features, how='inner').drop(columns=[header_features])
        return X
    
    return combine_metadata_and_features(features_df=features_df,
                                         metadata_df=metadata_df)

