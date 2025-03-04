import os
from pathlib import Path

# External packages
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import typer
from datetime import datetime

# Local packages
import feature_functions as feature_functions


# --- Read images off local ---

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

        if (img is not None) & (img.mode == 'RGB'):
                images.append(img)
            
    return filenames, images


def read_segmentations():
    import pickle
    p_path = '/Users/ntin/Models/sam2/notebooks/'
    corrected_segmentations = '2025-02-23_Hand_Segmentations-Corrected-3.pkl'

    with open(p_path+corrected_segmentations, 'rb') as file:
        segmentations = pickle.load(file)

    return segmentations


# --- Run app ---

def main(input_feature: str,
         src_dir: Path,
         dst_dir: Path) -> None:
    
    # --- Check arguments
    if not os.path.isdir(src_dir):
        typer.echo(f"src_dir '{src_dir}' not found", err=True)
        raise typer.Abort()
    
    if input_feature not in feature_functions.find_function:
        typer.echo(f"Could not identify input feature '{input_feature}'", err=True)
        raise typer.Abort()

    # --- Preparations
    today = datetime.today().strftime('%Y-%m-%d')
    experiment_name = f"{today}_hawkeye-hands-{input_feature}-features.csv"

    # Prepare destination directory
    os.makedirs(dst_dir, exist_ok=True)

    #  --- Load in images
    hawkeye_filenames, hawkeye_hands_images = read_local(src_dir)
    N_images = len(hawkeye_hands_images)
    assert N_images > 0 

    #  --- Load in segmentations
    segmentations = read_segmentations()

    X = []

    for fname, hand in tqdm(zip(hawkeye_filenames, hawkeye_hands_images)):
        key = fname.split('.')[0]
        mask = segmentations[key]
        masked_hand = np.where(mask[..., None], hand, 0)
        computed_features = feature_functions.feature_delegation(input_feature,masked_hand)
        X += [computed_features]
    
    computed_features = pd.DataFrame(X)
    computed_features.loc[:,'filename'] = hawkeye_filenames
    feature_path = os.path.join(dst_dir, experiment_name)
    computed_features.to_csv(feature_path)


# --- Run app ---

if __name__ == "__main__":
    typer.run(main) 

# poetry run python3 bin/calculate_hand.py lbp_glcm /Users/ntin/Documents/DermaML_local/hawkeye-hands-2024-07-29/images_processed /Users/ntin/Documents/DermaML_local/hawkeye-hands-2024-07-29/features-2025-02-24