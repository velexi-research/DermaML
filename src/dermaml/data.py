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
import math
import os
from pathlib import Path
from typing import List

# External packages
import cv2
import numpy as np
from numpy.random import default_rng
import skimage
from PIL import Image
from rembg import remove
import mediapipe as mp


# --- Public functions

def remove_alpha_channel(image: np.ndarray) -> np.ndarray:
    """
    Remove alpha channel from image.

    Parameters
    ----------
    image: NumPy array containing image. The array is expected to be arranged
        such that the right-most dimension specifies the color channel in the
        following order: R, G, B, A (if present)

    Return value
    ------------
    image_out: NumPy array containing image with alpha channel removed. The
        array is arranged such that the right-most dimension specifies the
        color channel:

        * image_out[:,:,0] contains the red channel

        * image_out[:,:,1] contains the green channel

        * image_out[:,:,2] contains the blue channel
    """
    # Remove alpha channel (if present)
    if image.shape[-1] == 4:
        return image[:, :, 0:-1]

    return image

def remove_bg(image):
    """
    Remove green background from images

    Paramerers
    ___________
    image: Opened image file

    Return value
    ____________
    output: Numpy array containing image with background removed
    """
    
    #Remove green screen background
    output = remove(image)

    #Return numpy array of image cutout
    return np.array(output)


# def remove_background(image: np.ndarray,
#                       lower_threshold: List = (25, 75, 85),
#                       upper_threshold: List = (130, 255, 190)) -> np.ndarray:
#     """
#     Remove green background from image.

#     Parameters
#     ----------
#     image: NumPy array containing image. The array is expected to be arranged
#         such that the right-most dimension specifies the color channel in the
#         following order: R, G, B, A (if present)

#     lower_threshold: (R, G, B) value to use as lower threshold for identifying
#         green pixels

#     upper_threshold: (R, G, B) value to use as upper threshold for identifying
#         green pixels

#     Return value
#     ------------
#     image_out: NumPy array containing image with background removed. The
#         array is arranged such that the right-most dimension specifies the
#         color channel:

#         * image_out[:,:,0] contains the red channel

#         * image_out[:,:,1] contains the green channel

#         * image_out[:,:,2] contains the blue channel
#     """
#     # --- Check arguments

#     # Convert color values in the interval [0, 255) with type 'int64'
#     if image.dtype in ['float32', 'float64']:
#         if np.max(image) >= 1:
#             image = (255*image).astype('int64')

#     # Remove alpha channel
#     image = remove_alpha_channel(image)

#     # --- Remove background

#     image_out = image.copy()
#     mask = cv2.inRange(image_out, lower_threshold, upper_threshold)
#     image_out[mask != 0] = [0, 0, 0]

#     return image_out


def generate_synthetic_dataset(image_path: Path,
                               dst_dir: Path,
                               size: int = 10,
                               width: int = 256, height: int = 256) -> List:
    """
    Generate synthetic dataset from the source image.

    Parameters
    ----------
    image_path: path to source image

    dst_dir: directory that synthetic image dataset should be saved to.
        _Note_: `dst_dir` should exist before this function is called.

    size: number of synthetic images to generate

    width: width of images in synthetic dataset

    height: height of images in synthetic dataset

    Return value
    ------------
    synthetic_images: filenames of synthetic images generated from the source
        image
    """
    # --- Check arguments

    if not os.path.isfile(image_path):
        raise ValueError(f"`image_path` {image_path} not found")

    if not os.path.isdir(dst_dir):
        raise ValueError(f"`dst_dir` {dst_dir} not found")

    if size <= 0:
        raise ValueError("`size` must be positive")

    if width <= 0:
        raise ValueError("`width` must be positive")

    if height <= 0:
        raise ValueError("`height` must be positive")

    # --- Preparations

    # Load image
    src_image = skimage.io.imread(image_path)

    # Check compatibility of image size with width and height arguments
    if width >= src_image.shape[1]:
        raise RuntimeError("`width` must be less than source image width")

    if height >= src_image.shape[0]:
        raise RuntimeError("`height` must be less than source image height")

    # Determine if source image is color or grayscale
    is_color = len(src_image.shape) > 2

    # Compute zero-padding size
    padding_size = math.floor(math.log(size) / math.log(10)) + 1

    # Initialize return value
    synthetic_images = []

    # --- Generate synthetic dataset

    # Pick random locations for top-left corner of sub-image
    rng = default_rng()
    rows = ((height // 2) * rng.random((size, 1))).astype('int')
    cols = ((width // 2) * rng.random((size, 1))).astype('int')
    indices = np.hstack((rows, cols))

    for k in range(size):
        # Extract sub-image
        i, j = indices[k, :]
        if is_color:
            image_out = src_image[i:i + height, j:j + width, :]
        else:
            image_out = src_image[i:i + height, j:j + width]

        # Save sub-image
        basename = os.path.basename(image_path)
        image_id = str(k+1).zfill(padding_size)
        filename = f"{os.path.splitext(basename)[0]}-{image_id}.png"
        output_path = os.path.join(dst_dir, filename)
        skimage.io.imsave(output_path, image_out)

        # Save filename
        synthetic_images.append(filename)

    return synthetic_images



def crop_palm(image):

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:

        # Read image file
        image = cv2.flip(cv2.imread(image), 1)

        # Convert BGR image to RGB
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_hand_landmarks:
            print("No hands detected in the image.")
        else:
            image_height, image_width, _ = image.shape
            annotated_image = image.copy()

            # Get first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]

            landmark_0 = (
                int(hand_landmarks.landmark[0].x * image_width),
                int(hand_landmarks.landmark[0].y * image_height)
            )

            landmark_1 = (
                int(hand_landmarks.landmark[1].x * image_width),
                int(hand_landmarks.landmark[1].y * image_height)
            )

            landmark_2 = (
                int(hand_landmarks.landmark[2].x * image_width),
                int(hand_landmarks.landmark[2].y * image_height)
            )

            landmark_5 = (
                int(hand_landmarks.landmark[5].x * image_width),
                int(hand_landmarks.landmark[5].y * image_height)
            )

            landmark_9 = (
                int(hand_landmarks.landmark[9].x * image_width),
                int(hand_landmarks.landmark[9].y * image_height)
            )

            landmark_13 = (
                int(hand_landmarks.landmark[13].x * image_width),
                int(hand_landmarks.landmark[13].y * image_height)
            )

            landmark_17 = (
                int(hand_landmarks.landmark[17].x * image_width),
                int(hand_landmarks.landmark[17].y * image_height)
            )

            # Draw circles for landmarks
            cv2.circle(annotated_image, landmark_0, 5, (0, 0, 255), -1) 
            cv2.circle(annotated_image, landmark_1, 5, (0, 0, 255), -1)  
            cv2.circle(annotated_image, landmark_2, 5, (0, 0, 255), -1)
            cv2.circle(annotated_image, landmark_5, 5, (0, 0, 255), -1) 
            cv2.circle(annotated_image, landmark_9, 5, (0, 0, 255), -1) 
            cv2.circle(annotated_image, landmark_13, 5, (0, 0, 255), -1)  
            cv2.circle(annotated_image, landmark_17, 5, (0, 0, 255), -1)  

            landmark_coordinates = [landmark_0, landmark_1, landmark_2, landmark_5, landmark_9, landmark_13, landmark_17]

            for i in range(len(landmark_coordinates) - 1):
                cv2.line(annotated_image, landmark_coordinates[i], landmark_coordinates[i + 1], (0, 0, 255), 2)

            # Connect the last landmark to the first landmark to complete the loop
            cv2.line(annotated_image, landmark_coordinates[-1], landmark_coordinates[0], (0, 0, 255), 2)

            # Create a mask of the region within the loop
            mask = np.zeros_like(image)
            cv2.fillPoly(mask, [np.array(landmark_coordinates)], (255, 255, 255))

            # Apply the mask to the original image to crop the region
            cropped_image = cv2.bitwise_and(image, mask)

            colored_image = cropped_image[:,:,::-1]

            if colored_image.shape[2] == 3:  # If the image is RGB
                resultant_image = cv2.cvtColor(colored_image, cv2.COLOR_RGB2RGBA)

            for i in range(resultant_image.shape[0]):
                for j in range(resultant_image.shape[1]):
                    # Check if pixel is black
                    if all(resultant_image[i, j, :3] == [0, 0, 0]):
                        # Set the alpha channel to 0 to make it transparent
                        resultant_image[i, j, 3] = 0

            return resultant_image

