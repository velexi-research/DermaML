#!/usr/bin/env python
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
"""
Script for preparing raw image data for feature extraction.
"""

# --- Imports

# Standard library
import os
from pathlib import Path

# External packages
import cv2
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import typer

# Local packages

experiment_name = "2024-08-02_hawkeye-hands-redness-features.csv"

def _engineer_redness(img):
    
    original_image = np.array(img)
    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)

    red_channel = rgb_image[:,:,0]

    engineered_features = {}

    # relative redness
    relative_red_mean, relative_red_std = np.mean(red_channel), np.std(red_channel)
    red_labels = ['relative_redness_mean', 'relative_redness_std']
    red_values = [relative_red_mean, relative_red_std]
    red_features = {k:v for k,v in zip(red_labels, red_values)}


    # feature engineering update
    engineered_features.update(red_features)

    return engineered_features


def read_local(image_dir, image_fnames=[]):
    '''
    Read in images from hawkeye hands using PIL Image
    '''
    if not image_fnames:
        image_fnames = os.listdir(image_dir)

    images = []
    filenames = []
    for filename in tqdm(image_fnames):
    
        try:
            img = Image.open(os.path.join(image_dir, filename))
            filenames += [filename]

        except Image.UnidentifiedImageError:
            print(filename)
            continue

        if (img is not None) & (img.mode == 'RGBA'):
                images.append(img)
            
    return filenames, images


# --- Main program

def main(src_dir: Path,
         dst_dir: Path,) -> None:
    """
    Prepare raw image data for feature extraction.
    """

    # --- Check arguments

    if not os.path.isdir(src_dir):
        typer.echo(f"src_dir '{src_dir}' not found", err=True)
        raise typer.Abort()

    # --- Preparations

    # Prepare destination directory
    os.makedirs(dst_dir, exist_ok=True)

    # Load in images
    hawkeye_filenames, hawkeye_hands_images = read_local(src_dir)

    X = []

    for hand in tqdm(hawkeye_hands_images):
        hand_features = _engineer_redness(hand)
        X += [hand_features]

    
    hand_features = pd.DataFrame(X)
    hand_features.loc[:,'filename'] = hawkeye_filenames
    feature_path = os.path.join(dst_dir, experiment_name)
    hand_features.to_csv(feature_path)


# --- Run app

if __name__ == "__main__":
    typer.run(main) 