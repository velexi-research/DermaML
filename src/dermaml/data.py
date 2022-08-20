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

def remove_background(image: np.ndarray,
                      lower_threshold: List,
                      upper_threshold: List) -> np.ndarray:
    """
    Remove green background from image.

    Parameters
    ----------
    image: NumPy array containing image. The array is expected to be arranged
        such that

        * image[:,:,0] contains the red channel

        * image[:,:,1] contains the green channel

        * image[:,:,2] contains the blue channel

    lower_threshold: (R, G, B) value to use as lower threshold for identifying
        green pixels

    upper_threshold: (R, G, B) value to use as upper threshold for identifying
        green pixels

    Return value
    ------------
    image_out: NumPy array containing image with background removed. The array
        arranged such that

        * image_out[:,:,0] contains the red channel

        * image_out[:,:,1] contains the green channel

        * image_out[:,:,2] contains the blue channel
    """
    # --- Check arguments

    # Convert color values in the interval [0, 1) with type 'float32'
    if image.dtype == 'int64':
        image = (image/255).astype('float32')
    elif image.dtype == 'float64':
        image = image.astype('float32')

    # --- Remove background

    image_out = image.copy()
    mask = cv2.inRange(image_out, lower_threshold, upper_threshold)
    image_out[mask != 0] = [0, 0, 0]

    return image_out
