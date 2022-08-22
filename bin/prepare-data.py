#!/usr/bin/env python
"""
Script for preparing raw image data for feature extraction.
"""

# --- Imports

# Standard library
import glob
import os
from pathlib import Path
from typing import Optional

# External packages
import skimage
import typer

# Local packages
import dermaml
import dermaml.data


# --- Main program

def main(src_dir: Path,
         dst_dir: Path,
         image_type: Optional[str] = "all") -> None:
    """
    Prepare raw image data for feature extraction.
    """
    # --- Check arguments

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


# --- Run app

if __name__ == "__main__":
    typer.run(main)
