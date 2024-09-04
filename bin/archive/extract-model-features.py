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
Script for extracting ML model features from clean image data.
"""

# --- Imports

# Standard library
import glob
import json
import os
from pathlib import Path

# External packages
import pandas as pd
import skimage
import typer

# Local packages
import dermaml
import dermaml.features


# --- Main program

def main(src_dir: Path,
         dst_dir: Path,
         image_type: str = "png",
         src_metadata_file: Path = "metadata.csv") -> None:
    """
    Extract ML model features from clean image data.
    """
    # --- Check arguments

    if not os.path.isdir(src_dir):
        typer.echo(f"src_dir '{src_dir}' not found", err=True)
        raise typer.Abort()

    src_metadata_path = os.path.join(src_dir, src_metadata_file)
    if not os.path.isfile(src_metadata_path):
        typer.echo(
            f"src-metadata-file '{src_metadata_file}' not found in src_dir",
            err=True)
        raise typer.Abort()

    if image_type is None or image_type.lower() == "all":
        image_ext_list = ["gif", "jpeg", "jpg", "png", "tiff"]
    else:
        if image_type.lower() in ["jpeg", "jpg"]:
            image_ext_list = ["jpeg", "jpg"]
        else:
            image_ext_list = [image_type]

    # --- Preparations

    # Prepare destination directory
    os.makedirs(dst_dir, exist_ok=True)

    # Get list of image files
    image_paths = []
    for image_ext in image_ext_list:
        image_paths.extend(glob.glob(os.path.join(src_dir, f'*.{image_ext}')))

    # Read source metadata
    src_metadata = pd.read_csv(src_metadata_path, index_col="file")

    # Initialize metadata for feature dataset
    metadata_feature_dataset = []

    # --- Extract model features from image files

    for image_path in image_paths:
        # Load image
        image = skimage.io.imread(image_path)

        # Extract features
        features = dermaml.features.extract_features(image)

        # Save model features
        basename = os.path.basename(image_path)
        features_file = f"{os.path.splitext(basename)[0]}.json"
        output_path = os.path.join(dst_dir, features_file)
        with open(output_path, 'w') as file_:
            json.dump(features, file_)

        # Generate metadata
        target = src_metadata.at[os.path.basename(image_path), 'target']
        metadata_feature_dataset.append(
            {"file": features_file, "target": target})

    # --- Write metadata

    metadata_path = os.path.join(dst_dir, "metadata.csv")
    metadata_df = pd.DataFrame.from_records(metadata_feature_dataset,
                                            columns=["file", "target"])
    metadata_df.to_csv(metadata_path, index=False)


# --- Run app

if __name__ == "__main__":
    typer.run(main)
