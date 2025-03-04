import cv2
import numpy as np

def normalized_redness(img:np.array, mask:np.array):
    '''
    Takes in a color image + its 2D mask, returns the average redness in the masked area
    '''
    masked_color_img = np.where(mask[..., None], img, 0) # redundant step

    lab_img = cv2.cvtColor(masked_color_img, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab_img)
    L_mask, A_mask = L[mask],A[mask]
    normalized_redness = np.mean(A_mask)/np.mean(L_mask)

    return normalized_redness