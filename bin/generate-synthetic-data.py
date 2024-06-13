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
Script for generating synthetic data from source data.
"""

# --- Imports

# Standard library
import glob
import os
from pathlib import Path

# External packages
import pandas as pd
import typer

# Local packages
import dermaml
import dermaml.data


# --- Main program

def main(src_dir: Path,
         dst_dir: Path,
         image_type: str = "all",
         src_metadata_file: Path = "metadata.csv",
         size: int = 20) -> None:
    """
    Generate synthetic data from source data.

    The metadata file is expected to be a CSV file with the following columns:
    'file' and 'target'. For each record, the 'file' column should contain the
    source data file and the 'target' column should contain the class or
    numerical value to be predicted.
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

    # Initialize metadata for synthetic dataset
    metadata_synthetic = []

    # --- Generate synthetic dataset from images

    for image_path in image_paths:
        # Generate synthetic dataset for image
        images = dermaml.data.generate_synthetic_dataset(image_path,
                                                         dst_dir,
                                                         size=size)

        # Generate metadata for synthetic dataset
        target = src_metadata.at[os.path.basename(image_path), 'target']
        metadata_synthetic.extend([{"file": filename, "target": target}
                                   for filename in images])

    # --- Write metadata

    metadata_path = os.path.join(dst_dir, "metadata.csv")
    metadata_df = pd.DataFrame.from_records(metadata_synthetic,
                                            columns=["file", "target"])
    metadata_df.to_csv(metadata_path, index=False)


# --- Run app

if __name__ == "__main__":
    typer.run(main)
