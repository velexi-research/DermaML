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
#
"""
The dermaml.data module supports data pre- and post-processing.
"""

# --- Imports

# Standard library
import math
import os
from pathlib import Path
from typing import List

# External packages
import cv2
import numpy as np
from numpy.random import default_rng
import skimage


# --- Public functions

def remove_alpha_channel(image: np.ndarray) -> np.ndarray:
    """
    Remove alpha channel from image.

    Parameters
    ----------
    image: NumPy array containing image. The array is expected to be arranged
        such that the right-most dimension specifies the color channel in the
        following order: R, G, B, A (if present)

    Return value
    ------------
    image_out: NumPy array containing image with alpha channel removed. The
        array is arranged such that the right-most dimension specifies the
        color channel:

        * image_out[:,:,0] contains the red channel

        * image_out[:,:,1] contains the green channel

        * image_out[:,:,2] contains the blue channel
    """
    # Remove alpha channel (if present)
    if image.shape[-1] == 4:
        return image[:, :, 0:-1]

    return image


def remove_background(image: np.ndarray,
                      lower_threshold: List = (25, 75, 85),
                      upper_threshold: List = (130, 255, 190)) -> np.ndarray:
    """
    Remove green background from image.

    Parameters
    ----------
    image: NumPy array containing image. The array is expected to be arranged
        such that the right-most dimension specifies the color channel in the
        following order: R, G, B, A (if present)

    lower_threshold: (R, G, B) value to use as lower threshold for identifying
        green pixels

    upper_threshold: (R, G, B) value to use as upper threshold for identifying
        green pixels

    Return value
    ------------
    image_out: NumPy array containing image with background removed. The
        array is arranged such that the right-most dimension specifies the
        color channel:

        * image_out[:,:,0] contains the red channel

        * image_out[:,:,1] contains the green channel

        * image_out[:,:,2] contains the blue channel
    """
    # --- Check arguments

    # Convert color values in the interval [0, 255) with type 'int64'
    if image.dtype in ['float32', 'float64']:
        if np.max(image) >= 1:
            image = (255*image).astype('int64')

    # Remove alpha channel
    image = remove_alpha_channel(image)

    # --- Remove background

    image_out = image.copy()
    mask = cv2.inRange(image_out, lower_threshold, upper_threshold)
    image_out[mask != 0] = [0, 0, 0]

    return image_out


def generate_synthetic_dataset(image_path: Path,
                               dst_dir: Path,
                               size: int = 10,
                               width: int = 256, height: int = 256) -> None:
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
    None
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

    # --- Generate synthetic dataset

    # Pick random locations for top-left corner of sub-image
    rng = default_rng()
    rows = ((height // 2) * rng.random((size, 1))).astype('int')
    cols = ((width // 2) * rng.random((size, 1))).astype('int')
    indices = np.hstack((rows, cols))

    for k in range(size):
        # Extract sub-image
        i, j = indices[k, :]
        if is_color:
            image_out = src_image[i:i + height, j:j + width, :]
        else:
            image_out = src_image[i:i + height, j:j + width]

        # Save sub-image
        filename = os.path.basename(image_path)
        image_id = str(k+1).zfill(padding_size)
        output_path = os.path.join(
            dst_dir, f"{os.path.splitext(filename)[0]}-{image_id}.png")
        skimage.io.imsave(output_path, image_out)
