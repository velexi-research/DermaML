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
Script for preparing raw image data for feature extraction.
"""

# --- Imports

# Standard library
import glob
import os
from pathlib import Path
import shutil

# External packages
import skimage
import typer

# Local packages
import dermaml
import dermaml.data


# --- Main program

def main(src_dir: Path,
         dst_dir: Path,
         image_type: str = "all",
         src_metadata_file: Path = "metadata.csv") -> None:
    """
    Prepare raw image data for feature extraction.
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

    # --- Prepare image files for feature extraction.

    for image_path in image_paths:
        # Load image
        image = skimage.io.imread(image_path)

        # Remove background
        if len(image.shape) > 2:
            image = dermaml.data.remove_background(image)

        # Save image
        filename = os.path.basename(image_path)
        output_path = os.path.join(dst_dir,
                                   f"{os.path.splitext(filename)[0]}.png")
        skimage.io.imsave(output_path, image)

    # --- Copy metadata file to dst_dir

    shutil.copy(src_metadata_path, dst_dir)


# --- Run app

if __name__ == "__main__":
    typer.run(main)
