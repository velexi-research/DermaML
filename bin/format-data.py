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
Script for transforming data into JPEG
"""

# --- Imports

# Standard library
import os
from pathlib import Path
import shutil

# External packages
import pandas as pd
import numpy as np
import skimage.io
import typer
from PIL import Image
import pillow_heif


# --- Main program

def main(src_dir: Path ,
         dst_dir: Path ,
         image_type: str = "all",
         src_metadata_file: Path = "metadata.csv") -> None:
    """
    Transform image data to JPEG if any images are in HEIC format
    """
    if not os.path.isdir(src_dir):
        typer.echo(f"src_dir '{src_dir}' not found", err=True)
        raise typer.Abort()

    src_metadata_path = os.path.join(src_dir, src_metadata_file)
    if not os.path.isfile(src_metadata_path):
        typer.echo(
            f"src-metadata-file '{src_metadata_file}' not found in src_dir",
            err=True)
        raise typer.Abort()


    # --- Preparations

    # Prepare destination directory
    os.makedirs(dst_dir, exist_ok=True)

    metadata = pd.read_csv(src_metadata_path)
    valid_image_files_df = metadata.loc[:,['left_hand_image_file', 'right_hand_image_file']]
    valid_image_files = valid_image_files_df.to_numpy().flatten()
    valid_unique_image_files = np.unique(valid_image_files)

    # Get list of image files

    src = str(src_dir)+'/'
    image_paths = [src + path for path in valid_unique_image_files]

    for image_path in image_paths:
        # Load images
        filename = os.path.basename(image_path)
        output_path = os.path.join(dst_dir,
                                   f"{os.path.splitext(filename)[0]}.jpeg")
                            
        try:
            #JPEG load if possible
            image = skimage.io.imread(image_path)
            skimage.io.imsave(output_path, image)

        except OSError:
            #HEIC load and convert to JPEG
            print(image_path)
            heif_file = pillow_heif.read_heif(image_path)
            image = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
)
            image.save(output_path, format("jpeg"))
            continue;
    # --- copy meta data 
    shutil.copy(src_metadata_path, dst_dir)

# --- Run app

if __name__ == "__main__":
    typer.run(main) 