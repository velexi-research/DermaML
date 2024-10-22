# --- External packages
import cv2
import pyfeats
import numpy as np
import dermaml.features as features

# --- File Index ---
# 0. Feature Delegation
# 1. Redness
# 2. LBP + GLCM
# 3. Hessian
# 4. Pyfeats Tectures

'''
find_function = {
    'redness': _redness,
    'lbp_glcm': _lbp_glcm, 
    'hessian': _hessian,
    'pyfeats': _pyfeats,
}
'''

# --- Feature Delegation ---

def feature_delegation(input_feat, hand_image):
    engineered_features = {}
    original_image = np.array(hand_image)
    mask = cv2.cvtColor(original_image, cv2.COLOR_RGBA2GRAY) != 0    

    if input_feat == 'redness':
        red_channel = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)[:,:,0]
        features = find_function[input_feat](red_channel)

    else:   
        hsv_image = features.remove_brightness(hand_image)

        if input_feat == 'lbp_glcm':
            features = find_function[input_feat](hsv_image)
        
        else:
            bw_image = hsv_image[:,:,1]

            if input_feat == 'pyfeats':
                features = find_function[input_feat](bw_image, mask)

            elif input_feat == 'hessian':
                hessian_image_a, _ = features.detect_ridges(bw_image) 
                features = find_function[input_feat](hessian_image_a, mask)

            else:
                assert "Could not identify input feature. Please try again."
    
    engineered_features.update(features)
    return engineered_features


# --- Feature Computations --

def _redness(red_channel):
    # relative redness
    relative_red_mean, relative_red_std = np.mean(red_channel), np.std(red_channel)
    red_labels = ['relative_redness_mean', 'relative_redness_std']
    red_values = [relative_red_mean, relative_red_std]
    red_features = {k:v for k,v in zip(red_labels, red_values)}

    return red_features



def _lbp_glcm(hsv_image):
    # lbp_
    lbp_hist,lbp = features.compute_lbp(hsv_image)
    enum_lbp = dict(enumerate(lbp_hist))
    lbp_features = {'lbp_'+str(k):v for k,v in enum_lbp.items()}

    # glcm_whole_image
    contrast, correlation, energy, homogeneity = features.compute_glcm(hsv_image)
    glcm_labels = ['contrast', 'correlation', 'energy', 'homogeneity']
    glcm_values = [contrast[0][0], correlation[0][0], energy[0][0], homogeneity[0][0]]
    glcm_scikit_features = {str(k)+'_scikit':v for k,v in zip(glcm_labels, glcm_values)}

    return lbp_features, glcm_scikit_features



def _hessian(hessian_image_a, mask):
    # unnormalized hessian ridges (wrinkles)
    a_lim = lambda a : np.mean(a) + 2*(np.std(a))
    mask_area = np.count_nonzero(mask)
    hessian_ridges = hessian_image_a >= a_lim(hessian_image_a)
    hessian_ridge_value = np.count_nonzero(hessian_ridges)
    hessian_values = [hessian_ridge_value, hessian_ridge_value/mask_area]
    hessian_label = ['skin_folds_hessian', 'skin_folds_hessian_pct_mask']
    hessian_features = {k:v for k,v in zip(hessian_label, hessian_values)}

    return hessian_features


def _pyfeats(bw_image, mask):
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

    return glcm_pyfeats_features, glds_features, ngtdm_features, lte_features



# --- Mapping

find_function = {
    'redness': _redness,
    'lbp_glcm': _lbp_glcm, 
    'hessian': _hessian,
    'pyfeats': _pyfeats,
}