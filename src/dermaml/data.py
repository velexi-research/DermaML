"""
The dermaml.data module provides data utilities
"""

# --- Imports

# Standard library
import math
import os
from pathlib import Path
from typing import List

# External packages
import numpy as np
from numpy.random import default_rng
import skimage


# --- Public functions


def generate_synthetic_dataset(
    image_path: Path, dst_dir: Path, size: int = 10, width: int = 256, height: int = 256
) -> List:
    """
    Generate synthetic dataset from the source image.

    Parameters
    ----------
    image_path: path to source image

    dst_dir: directory that synthetic image dataset should be saved to.
        _Note_: `dst_dir` should exist before this function is called.

    size: number of synthetic images to generate

    width: width of images in synthetic dataset

    height: height of images in synthetic dataset

    Return value
    ------------
    synthetic_images: filenames of synthetic images generated from the source
        image
    """
    # --- Check arguments

    if not os.path.isfile(image_path):
        raise ValueError(f"`image_path` {image_path} not found")

    if not os.path.isdir(dst_dir):
        raise ValueError(f"`dst_dir` {dst_dir} not found")

    if size <= 0:
        raise ValueError("`size` must be positive")

    if width <= 0:
        raise ValueError("`width` must be positive")

    if height <= 0:
        raise ValueError("`height` must be positive")

    # --- Preparations

    # Load image
    src_image = skimage.io.imread(image_path)

    # Check compatibility of image size with width and height arguments
    if width >= src_image.shape[1]:
        raise RuntimeError("`width` must be less than source image width")

    if height >= src_image.shape[0]:
        raise RuntimeError("`height` must be less than source image height")

    # Determine if source image is color or grayscale
    is_color = len(src_image.shape) > 2

    # Compute zero-padding size
    padding_size = math.floor(math.log(size) / math.log(10)) + 1

    # Initialize return value
    synthetic_images = []

    # --- Generate synthetic dataset

    # Pick random locations for top-left corner of sub-image
    rng = default_rng()
    rows = ((height // 2) * rng.random((size, 1))).astype("int")
    cols = ((width // 2) * rng.random((size, 1))).astype("int")
    indices = np.hstack((rows, cols))

    for k in range(size):
        # Extract sub-image
        i, j = indices[k, :]
        if is_color:
            image_out = src_image[i : i + height, j : j + width, :]
        else:
            image_out = src_image[i : i + height, j : j + width]

        # Save sub-image
        basename = os.path.basename(image_path)
        image_id = str(k + 1).zfill(padding_size)
        filename = f"{os.path.splitext(basename)[0]}-{image_id}.png"
        output_path = os.path.join(dst_dir, filename)
        skimage.io.imsave(output_path, image_out)

        # Save filename
        synthetic_images.append(filename)

    return synthetic_images
