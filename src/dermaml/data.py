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
from typing import List

# External packages
import cv2
import numpy as np


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
