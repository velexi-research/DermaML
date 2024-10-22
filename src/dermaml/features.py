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
from dermaml import image
import pyfeats
from skimage.feature import hessian_matrix, hessian_matrix_eigvals


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


def compute_lbp(image: np.ndarray, radius=3, num_points=None,) -> np.ndarray:
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
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the co-occurrence matrix for the image
    co_matrix = skimage.feature.graycomatrix( #FIXME : choice of kernel size
        image, [5], [0], levels=256, symmetric=True, normed=True
    )

    # Calculate texture features from the co-occurrence matrix
    contrast = skimage.feature.graycoprops(co_matrix, "contrast")
    correlation = skimage.feature.graycoprops(co_matrix, "correlation")
    energy = skimage.feature.graycoprops(co_matrix, "energy")
    homogeneity = skimage.feature.graycoprops(co_matrix, "homogeneity")

    return contrast, correlation, energy, homogeneity


## -- Updated: Feature Computations

# Image transformations
def detect_ridges(gray, sigma=1.0):
  '''
  FIXME
  '''
  H_elems = hessian_matrix(gray, sigma=sigma)
  maxima_ridge, minima_ridge = hessian_matrix_eigvals(H_elems)
  return maxima_ridge, minima_ridge

def _engineer_features(img):
    
    # original_image = image.remove_background(img)

    original_image = np.array(img)
    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)
    hsv_image = image.remove_brightness(rgb_image)
    bw_image = hsv_image[:,:,1]
    mask = cv2.cvtColor(rgb_image, cv2.COLOR_RGBA2GRAY) != 0 
    hessian_image_a, hessian_image_b = detect_ridges(bw_image) 
    red_channel = rgb_image[:,:,0]

    engineered_features = {}

    # relative redness
    relative_red_mean, relative_red_std = np.mean(red_channel), np.std(red_channel)
    red_labels = ['relative_redness_mean', 'relative_redness_std']
    red_values = [relative_red_mean, relative_red_std]
    red_features = {k:v for k,v in zip(red_labels, red_values)}


    # lbp_
    lbp_hist,lbp = compute_lbp(hsv_image)
    enum_lbp = dict(enumerate(lbp_hist))
    lbp_features = {'lbp_'+str(k):v for k,v in enum_lbp.items()}


    # glcm_whole_image
    contrast, correlation, energy, homogeneity = compute_glcm(hsv_image)
    glcm_labels = ['contrast', 'correlation', 'energy', 'homogeneity']
    glcm_values = [contrast[0][0], correlation[0][0], energy[0][0], homogeneity[0][0]]
    glcm_scikit_features = {str(k)+'_scikit':v for k,v in zip(glcm_labels, glcm_values)}


    # (pyfeats) glcm
    features_mean, features_range, labels_mean, labels_range = pyfeats.glcm_features(bw_image, ignore_zeros=True)
    glcm_pyfeats_features = {str(k)+'_pyfeats':v for k,v in zip(labels_mean, features_mean)}


    # glds
    glds_values, glds_labels = pyfeats.glds_features(bw_image, mask, Dx=[0,1,1,1], Dy=[1,1,0,-1])
    glds_features = {str(k)+'_glds':v for k,v in zip(glds_labels, glds_values)}


    # ngtdm
    ngtdm_values, ngtdm_labels = pyfeats.ngtdm_features(bw_image, mask, d=1)
    ngtdm_features = {str(k)+'_ngtdm':v for k,v in zip(ngtdm_labels, ngtdm_values)}

    
    # lte
    lte_values, lte_labels, = pyfeats.lte_measures(bw_image, mask,)
    lte_features = {str(k)+'_lte':v for k,v in zip(lte_labels, lte_values)}


    # unnormalized hessian ridges (wrinkles)
    a_lim = lambda a : np.mean(a) + 2*(np.std(a))
    mask_area = np.count_nonzero(mask)

    hessian_ridges = hessian_image_a >= a_lim(hessian_image_a)
    hessian_ridge_value = np.count_nonzero(hessian_ridges)
    hessian_values = [hessian_ridge_value, hessian_ridge_value/mask_area]
    hessian_label = ['skin_folds_hessian', 'skin_folds_hessian_pct_mask']
    hessian_features = {k:v for k,v in zip(hessian_label, hessian_values)}


    # feature engineering update
    engineered_features.update(red_features)
    engineered_features.update(lbp_features)
    engineered_features.update(glcm_scikit_features)
    engineered_features.update(glcm_pyfeats_features)
    engineered_features.update(glds_features)
    engineered_features.update(ngtdm_features)
    engineered_features.update(lte_features)
    engineered_features.update(hessian_features)

    return engineered_features


