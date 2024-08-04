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
The dermaml.image module provides image processing functionality.
"""

# --- Imports

# Standard library
import io
import os
from pathlib import Path
from typing import Union

# External packages
import cv2
import numpy as np
from PIL import Image
import rembg
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
        output = image[:, :, 0:-1]
    else:
        output = image

    return output


def remove_background(image: np.ndarray) -> np.ndarray:
    """
    Remove background from `image`.

    Paramerers
    ___________
    image: image data

    Return value
    ____________
    output: Numpy array containing image with background removed
    """
    # --- Check arguments

    if not isinstance(image, (np.ndarray)):
        raise TypeError(
            "`image` must be of type `np.ndarray`. "
            + f"(type(image)={type(image)}"
        )

    # --- Remove background
    #
    # Note: the return type of rembg.remove() is the same as the type of `image`
    cutout = rembg.remove(image)

    # Return numpy array representation of image with background removed
    if isinstance(cutout, np.ndarray):
        output = cutout

    else:
        print('Not sure how you got here')

    return output


def remove_brightness(image):
    '''
    Converts an RGB-channeled image to HSV/HSB and removes the 'value' or 'brightness' channel.

    Arguments
    ---------
    `image`: an RGB numpy array
    '''
    assert len(image.shape) == 3
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_image[:, :, 2] = 0
    return hsv_image



def crop_palm(image_path: Path) -> Image:
    """
    TODO
    """

    mp_hands = mp.solutions.hands

    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.01
    ) as hands:

        # Read image file
        image = cv2.flip(cv2.imread(image_path), 1)

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
                int(hand_landmarks.landmark[0].y * image_height),
            )

            landmark_1 = (
                int(hand_landmarks.landmark[1].x * image_width),
                int(hand_landmarks.landmark[1].y * image_height),
            )

            landmark_2 = (
                int(hand_landmarks.landmark[2].x * image_width),
                int(hand_landmarks.landmark[2].y * image_height),
            )

            landmark_5 = (
                int(hand_landmarks.landmark[5].x * image_width),
                int(hand_landmarks.landmark[5].y * image_height),
            )

            landmark_9 = (
                int(hand_landmarks.landmark[9].x * image_width),
                int(hand_landmarks.landmark[9].y * image_height),
            )

            landmark_13 = (
                int(hand_landmarks.landmark[13].x * image_width),
                int(hand_landmarks.landmark[13].y * image_height),
            )

            landmark_17 = (
                int(hand_landmarks.landmark[17].x * image_width),
                int(hand_landmarks.landmark[17].y * image_height),
            )

            # Draw circles for landmarks
            cv2.circle(annotated_image, landmark_0, 5, (0, 0, 255), -1)
            cv2.circle(annotated_image, landmark_1, 5, (0, 0, 255), -1)
            cv2.circle(annotated_image, landmark_2, 5, (0, 0, 255), -1)
            cv2.circle(annotated_image, landmark_5, 5, (0, 0, 255), -1)
            cv2.circle(annotated_image, landmark_9, 5, (0, 0, 255), -1)
            cv2.circle(annotated_image, landmark_13, 5, (0, 0, 255), -1)
            cv2.circle(annotated_image, landmark_17, 5, (0, 0, 255), -1)

            landmark_coordinates = [
                landmark_0,
                landmark_1,
                landmark_2,
                landmark_5,
                landmark_9,
                landmark_13,
                landmark_17,
            ]

            for i in range(len(landmark_coordinates) - 1):
                cv2.line(
                    annotated_image,
                    landmark_coordinates[i],
                    landmark_coordinates[i + 1],
                    (0, 0, 255),
                    2,
                )

            # Connect the last landmark to the first landmark to complete the loop
            cv2.line(
                annotated_image,
                landmark_coordinates[-1],
                landmark_coordinates[0],
                (0, 0, 255),
                2,
            )

            # Create a mask of the region within the loop
            mask = np.zeros_like(image)
            cv2.fillPoly(mask, [np.array(landmark_coordinates)], (255, 255, 255))

            # Apply the mask to the original image to crop the region
            cropped_image = cv2.bitwise_and(image, mask)

            colored_image = cropped_image[:, :, ::-1]

            if colored_image.shape[2] == 3:  # If the image is RGB
                resultant_image = cv2.cvtColor(colored_image, cv2.COLOR_RGB2RGBA)

            for i in range(resultant_image.shape[0]):
                for j in range(resultant_image.shape[1]):
                    # Check if pixel is black
                    if all(resultant_image[i, j, :3] == [0, 0, 0]):
                        # Set the alpha channel to 0 to make it transparent
                        resultant_image[i, j, 3] = 0

            return resultant_image


def multi_crop_palm(image_path: Path) -> dict:
    """
    TODO
    """

    mp_hands = mp.solutions.hands

    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.01
    ) as hands:

        # Get filename from the image path
        image_filename = os.path.basename(image_path)

        # Read image file
        image = cv2.flip(cv2.imread(image_path), 1)

        # Convert BGR image to RGB
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_hand_landmarks:
            print("No hands detected in the image.")
            return None

        image_height, image_width, _ = image.shape
        annotated_image = image.copy()

        # Get first hand
        hand_landmarks = results.multi_hand_landmarks[0]

        landmark_9 = (
            int(hand_landmarks.landmark[9].x * image_width),
            int(hand_landmarks.landmark[9].y * image_height),
        )

        landmark_10 = (
            int(hand_landmarks.landmark[10].x * image_width),
            int(hand_landmarks.landmark[10].y * image_height),
        )

        # Define rectangle thickness
        rect_thickness = 2

        # Draw rectangles around landmarks 9 and 10
        rect_color = (0, 0, 255)
        rect_size = 40

        cv2.rectangle(
            annotated_image,
            (landmark_9[0] - rect_size, landmark_9[1] - rect_size),
            (landmark_9[0] + rect_size, landmark_9[1] + rect_size),
            rect_color,
            rect_thickness,
        )

        cv2.rectangle(
            annotated_image,
            (landmark_10[0] - rect_size, landmark_10[1] - rect_size),
            (landmark_10[0] + rect_size, landmark_10[1] + rect_size),
            rect_color,
            rect_thickness,
        )

        # Crop images inside landmarks 9 and 10
        cropped_image_9 = annotated_image[
            landmark_9[1]
            - rect_size
            + rect_thickness : landmark_9[1]
            + rect_size
            - rect_thickness,
            landmark_9[0]
            - rect_size
            + rect_thickness : landmark_9[0]
            + rect_size
            - rect_thickness,
        ]

        cropped_image_10 = annotated_image[
            landmark_10[1]
            - rect_size
            + rect_thickness : landmark_10[1]
            + rect_size
            - rect_thickness,
            landmark_10[0]
            - rect_size
            + rect_thickness : landmark_10[0]
            + rect_size
            - rect_thickness,
        ]

        # Create a dictionary for the results of this single image
        cropped_dict = {"Image 9": cropped_image_9, "Image 10": cropped_image_10}

        # Create a dictionary with the image filename as the main key
        final_dict = {image_filename: cropped_dict}

        return final_dict
