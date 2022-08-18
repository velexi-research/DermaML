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

# External packages
import numpy as np


# --- Public functions

def remove_background(
        R_in: np.ndarray, G_in: np.ndarray, B_in: np.ndarray) -> tuple:
    """
    Remove green background from image.

    Parameters
    ----------
    R_in: red channel of original image

    G_in: green channel of original image

    B_in: blue channel of original image

    Return value
    ------------
    R_out: red channel of image with background removed

    G_out: green channel of image with background removed

    B_out: blue channel of image with background removed
    """
    # --- Check arguments

    # Convert color values in the interval [0, 1) with type 'float32'

    if R_in.dtype == 'float64':
        R_in = R_in.astype('float32')
    elif np.issubdtype(R_in.dtype, np.integer):
        R_in = R_in.astype('float32') / 255

    if G_in.dtype == 'float64':
        G_in = G_in.astype('float32')
    elif np.issubdtype(G_in.dtype, np.integer):
        G_in = G_in.astype('float32') / 255

    if B_in.dtype == 'float64':
        B_in = B_in.astype('float32')
    elif np.issubdtype(B_in.dtype, np.integer):
        B_in = B_in.astype('float32') / 255

    # --- Remove background

    # TODO
    R_out = R_in.copy()
    G_out = G_in.copy()
    B_out = B_in.copy()

    return (R_out, G_out, B_out)
