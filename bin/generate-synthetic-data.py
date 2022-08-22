#!/usr/bin/env python
"""
Script for generating synthetic data from source data.
"""

# --- Imports

# Standard library
import glob
import os
from pathlib import Path
from typing import Optional

# External packages
import typer

# Local packages
import dermaml
import dermaml.data


# --- Main program

def main(src_dir: Path,
         dst_dir: Path,
         image_type: Optional[str] = "all",
         size: int = 20) -> None:
    """
    Generate synthetic data from source data.
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

    # --- Generate synthetic dataset from images

    for image_path in image_paths:
        # Generate synthetic dataset for image
        dermaml.data.generate_synthetic_dataset(image_path, dst_dir, size=size)


# --- Run app

if __name__ == "__main__":
    typer.run(main)
