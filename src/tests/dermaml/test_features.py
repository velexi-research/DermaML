"""
Unit tests for the `dermaml.features` module
"""

# --- Imports

# Standard library
import os

# Local imports
from dermaml import features
from PIL import Image
import numpy as np


# --- Tests


def test_compute_glcm():
    """
    Test compute_glcm().
    """
    # --- Preparations

    # Image filename
    image_filename = "glcm-test-image.png"

    # Create path to image file
    image_path = os.path.join(os.path.dirname(__file__), "data", image_filename)

    # Open image
    image = np.asarray(Image.open(image_path))

    # --- Test

    contrast, correlation, energy, homogeneity = features.compute_glcm(image)

    assert np.testing.assert_allclose(contrast, [[65.9359447]])
    assert np.testing.assert_allclose(correlation, [[0.97014729]])
    assert np.testing.assert_allclose(energy, [[0.02065778]])
    assert np.testing.assert_allclose(homogeneity, [[0.19573333]])
