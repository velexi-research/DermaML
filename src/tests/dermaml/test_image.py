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
Unit tests for the `dermaml.image` module
"""

# --- Imports

# Standard library
import os

# Local imports
import dermaml.image
from PIL import Image
import numpy as np
import pytest


# --- Tests


def test_remove_alpha_channel():
    """
    Test remove_alpha_channel().
    """
    # --- Preparations

    # Image filename
    image_filename = "remove-alpha-channel-test-image.png"

    # Create path to image file
    image_path = os.path.join(os.path.dirname(__file__), "data", image_filename)

    # Open image and convert it to a NumPy array
    image = np.asarray(Image.open(image_path))

    assert image.shape[-1] == 4

    # --- Test

    processed_image = dermaml.image.remove_alpha_channel(image)

    assert isinstance(processed_image, np.ndarray)
    assert processed_image.shape[-1] == 3


@pytest.mark.skip("TODO")
def test_remove_background():
    """
    Test remove_background().
    """
    # --- Preparations

    # Image filename
    # image_filename = "TODO.png"

    # Create path to image file
    # image_path = os.path.join(os.path.dirname(__file__), "data", image_filename)

    # Open image and convert it to a NumPy array
    # image = np.asarray(Image.open(image_path))

    # --- Test

    # TODO


@pytest.mark.skip("TODO")
def test_crop_palm():
    """
    Test crop_palm().
    """
    # --- Preparations

    # Image filename
    # image_filename = "TODO.png"

    # Create path to image file
    # image_path = os.path.join(os.path.dirname(__file__), "data", image_filename)

    # --- Test

    # TODO


@pytest.mark.skip("TODO")
def test_multi_crop_palm():
    """
    Test multi_crop_palm().
    """
    # --- Preparations

    # Image filename
    # image_filename = "TODO.png"

    # Create path to image file
    # image_path = os.path.join(os.path.dirname(__file__), "data", image_filename)

    # --- Test

    # TODO
