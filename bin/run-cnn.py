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
Script for running AutoML evaluation.
"""

# --- Imports

# Standard library
import csv
from pathlib import Path
import numpy as np
import pickle
import os

# External packages
import model_setup
import cv2 as cv
import typer
import yaml

# --- Read Segmentation File
def read_segmentation_file(segmentation_file:str):
    '''
    Read stored segmentations from SAM2 pipeline
    _______
    
    Returns: dictionary[filename] = arr
    '''
    # Check arguments
    if not os.path.exists(segmentation_file):
        typer.echo(f"segmentation_file '{segmentation_file}' not found", err=True)
        raise typer.Abort()
    
    # Read file
    with open(segmentation_file, 'rb') as file:
        segmentations = pickle.load(file)
    return segmentations


# --- Image Input Standardization

def square_resize(im, mask=None):
    '''
    Fit an image to a (512, 512) square by first padding the image
    _______
    
    Returns: Image of shape(512, 512, 3)
    '''

    if mask is not None:
        im = np.where(mask[..., None], im, 0)

    # calculate padding distance
    h, w, c = im.shape
    side = max(w, h)
    delta_w = side-w
    delta_h = side-h
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    # add 0 border
    square = cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_CONSTANT,value=0)

    # reshape to size (512, 512)
    norm_square = cv.resize(square, (512, 512))

    return norm_square


# --- Main program

def main(segmentation_file: Path = '/Users/ntin/Models/sam2/notebooks/2025-02-23_Hand_Segmentations-Corrected-3.pkl',
         metadata_file: Path = "metadata.csv",
         num_best: int = 5,
         experiment_name: str = "cnn",
         ) -> None:
    """
    Run AutoML evaluation.

    Results are output two files: 'model-scores.csv'
    """
    # --- Check arguments

    if num_best <= 0:
        typer.echo(
            "num-best must be strictly positive",
            err=True)
        raise typer.Abort()

    # --- Preparations

    masks = read_segmentation_file(segmentation_file)

    


    # --- Perform AutoML evaluation

    # Set up the dataset for CNN
    ...


# --- Run app

if __name__ == "__main__":
    typer.run(main)
