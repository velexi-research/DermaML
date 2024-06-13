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


@pytest.skip("BROKEN")
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
