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
The dermaml.features module supports feature extraction from images.
"""

# --- Imports

# Standard library

# External packages
import numpy as np
import skimage
import cv2


# --- Public functions


def extract_features(image: np.ndarray) -> dict:
    """
    Extract features from image.

    Parameters
    ----------
    image: NumPy array containing image. The array is expected to be arranged
        such that

        * image[:,:,0] contains the red channel

        * image[:,:,1] contains the green channel

        * image[:,:,2] contains the blue channel

    Return value
    ------------
    features: features extracted from image
    """
    # --- Check arguments

    # Convert color values in the interval [0, 1) with type 'float32'
    if image.dtype == "int64":
        image = (image / 255).astype("float32")
    elif image.dtype == "float64":
        image = image.astype("float32")

    # --- Preparations

    # Initialize features
    features = {}

    # --- Extract features

    # Compute texture histogram
    lbp_hist, _ = compute_lbp(image)
    enum_lbp = dict(enumerate(lbp_hist))
    lbp_dict = {'lbp_'+str(k):v for k,v in enum_lbp.items()}

    contrast, correlation, energy, homogeneity = compute_glcm(image)
    glcm_dict = {
                    'contrast':contrast[0][0], 
                     'correlation':correlation[0][0], 
                     'energy':energy[0][0], 
                     'homogeneity':homogeneity[0][0]
                     }

    
    return lbp_dict | glcm_dict


def compute_lbp(image: np.ndarray, radius=3, num_points=None) -> np.ndarray:
    """
    Compute local binary patterns (LBP) for image using "uniform" method.

    Parameters
    ----------
    image: grayscale image

    radius: radius of circle used to compute local binary patterns.

    num_points: number of points on circle used to compute local binary
        patterns. If num_points is set to None, (3 * radius) points are
        used to compute LBP values.

    Return values
    -------------
    lbp_hist: histogram of local binary pattern (LBP) values

    lbp: grayscale image with pixel values equal to LBP values
    """
    # --- Check arguments

    # Set num_points
    if num_points is None:
        num_points = 3 * radius

        # Convert pixel values to the interval [0, 1) with type 'integer'
    if image.dtype == "float32":
        if np.max(image) > 1:
            image = (255 * image).astype("int")

    # Transform image to grayscale
    if len(image.shape) > 2:
        image = skimage.color.rgb2gray(image)

    # --- Compute LBP image

    lbp = skimage.feature.local_binary_pattern(
        image, num_points, radius, method="uniform"
    )

    # --- Compute LBP histogram

    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, num_points + 3),
        range=(0, num_points + 2),
        density=True,
    )

    return lbp_hist, lbp


def compute_glcm(image: np.ndarray) -> tuple[float, float, float, float]:
    """
    Compute gray-level co-occcurence matrix (GLCM) for image.

    Parameters
    ----------
    image: image to perform GLCM on

    Return values
    -------------
    * contrast: measures the intensity contrast between a pixel and its neighbor over the
        whole image

    * correlation: measures how correlated a pixel is to its neighbor over the whole image

    * energy: returns the sum of squared elements in the GLCM

    * homogeneity: measures the closeness of the distribution of elements in the GLCM to
        the GLCM diagonal
    """
    # TODO: move color processing outside of function
    image_cv2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the co-occurrence matrix for the image
    co_matrix = skimage.feature.graycomatrix( #FIXME : choice of kernel size
        image_cv2, [5], [0], levels=256, symmetric=True, normed=True
    )

    # Calculate texture features from the co-occurrence matrix
    contrast = skimage.feature.graycoprops(co_matrix, "contrast")
    correlation = skimage.feature.graycoprops(co_matrix, "correlation")
    energy = skimage.feature.graycoprops(co_matrix, "energy")
    homogeneity = skimage.feature.graycoprops(co_matrix, "homogeneity")

    return contrast, correlation, energy, homogeneity
