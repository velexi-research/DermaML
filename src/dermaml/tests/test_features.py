#
#   Copyright 2022 Kevin Chu
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
Unit tests for `utils.py` module.
"""

# --- Imports

# Standard library
import os

# Local imports
from dermaml import features
from PIL import Image
import numpy as np
import pytest


# --- Tests

def test_compute_glcm():
    """
    Test utils.test_get_experiment_name().
    """
    # --- Preparations

    # Absolute path of the script
    script_path = os.path.abspath(__file__)

    # Image filename
    image_filename = "Screenshot 2023-09-12 at 4.30.47 PM.png"

    # Create absolute path to image file
    image_path = os.path.join(os.path.dirname(script_path), image_filename)

    # Open image
    generated_im = Image.open(image_path)

    # --- Test

    # contrast, correlation, energy, homogeneity = features.compute_glcm(im)

    contrast, correlation, energy, homogeneity = features.compute_glcm(generated_im)

    # assert contrast == np.array([[65.9359447]])

    # assert contrast == pytest.approx(65.9359447)
    # assert correlation == pytest.approx(0.97014729)
    # assert energy == pytest.approx(0.02065778)
    # assert homogeneity == pytest.approx(0.19573333)

    # # assert energy == np.array([[0.0001]])
    assert energy == pytest.approx(0.5) #obtained 0.58744519
    assert contrast == pytest.approx(0) #obtained 284.18194503
    assert homogeneity == pytest.approx(0.8750) #obtained 0.90707618
    assert correlation == pytest.approx(1) #obtained 0.96327682

    print("Contrast:", contrast)
    print("Correlation:", correlation)
    print("Energy:", energy)
    print("Homogeneity:", homogeneity)

