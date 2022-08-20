#!/usr/bin/env python
"""
Script for preparing raw image data for feature extraction and engineering.
"""

# --- Imports

# Standard library
import glob
import os
from pathlib import Path

# External packages
import skimage
import typer

# Local packages
import dermaml
import dermaml.data


# --- Main program

def main(src_dir: Path, dst_dir: Path, image_type: str = "jpeg") -> None:
    """
    Prepare raw image data for feature extraction and engineering.
    """
    # --- Preparations

    # Prepare destination directory
    os.makedirs(dst_dir, exist_ok=True)

    # Get list of image files
    image_files = []
    if image_type in ["jpeg", "jpg"]:
        image_files.extend(glob.glob(os.path.join(src_dir, '*.jpeg')))
        image_files.extend(glob.glob(os.path.join(src_dir, '*.jpg')))
    else:
        image_files.extend(glob.glob(os.path.join(src_dir, f'*.{image_type}')))

    # --- Pre-process image files

    for image_file in image_files:
        # Load image
        image = skimage.io.imread(image_file)

        # Remove background
        if len(image.shape) > 2:
            image = dermaml.data.remove_background(image)

        # Save image
        filename = os.path.basename(image_file)
        output_path = f"{os.path.splitext(filename)[0]}.png"
        image = skimage.io.imsave(os.path.join(dst_dir, output_path), image)


# --- Run app

if __name__ == "__main__":
    typer.run(main)
