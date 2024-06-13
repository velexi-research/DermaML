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
