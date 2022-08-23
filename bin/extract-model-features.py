#!/usr/bin/env python
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
import skimage
import typer

# Local packages
import dermaml
import dermaml.features


# --- Main program

def main(src_dir: Path,
         dst_dir: Path,
         image_type: str = "png") -> None:
    """
    Extract ML model features from clean image data.
    """
    # --- Check arguments

    if not os.path.isdir(src_dir):
        typer.echo(f"src_dir` '{src_dir}' not found", err=True)
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

    # --- Extract model features from image files

    for image_path in image_paths:
        # Load image
        image = skimage.io.imread(image_path)

        # Extract features
        features = dermaml.features.extract_features(image)

        # Save model features
        filename = os.path.basename(image_path)
        output_path = os.path.join(dst_dir,
                                   f"{os.path.splitext(filename)[0]}.json")

        with open(output_path, 'w') as file_:
            json.dump(features, file_)


# --- Run app

if __name__ == "__main__":
    typer.run(main)
