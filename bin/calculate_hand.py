import os
from pathlib import Path

# External packages
import pandas as pd
from PIL import Image
from tqdm import tqdm
import typer
from datetime import datetime

# Local packages
import bin.hand_features as hand_features

# --- Run app ---

def main(input_feature: str,
         src_dir: Path,
         dst_dir: Path) -> None:
    
    # --- Check arguments
    if not os.path.isdir(src_dir):
        typer.echo(f"src_dir '{src_dir}' not found", err=True)
        raise typer.Abort()
    
    if input_feature not in hand_features.find_function:
        typer.echo(f"Could not identify input feature '{input_feature}'", err=True)
        raise typer.Abort()

    # --- Preparations
    today = datetime.today().strftime('%Y-%m-%d')
    experiment_name = f"{today}_hawkeye-hands-{input_feature}-features.csv"

    # Prepare destination directory
    os.makedirs(dst_dir, exist_ok=True)

    #  --- Load in images
    hawkeye_filenames, hawkeye_hands_images = read_local(src_dir)
    # N_images = len(hawkeye_hands_images)

    X = []

    for hand in tqdm(hawkeye_hands_images):
        computed_features = hand_features.feature_delegation(input_feature,hand)
        X += [computed_features]
    
    computed_features = pd.DataFrame(X)
    computed_features.loc[:,'filename'] = hawkeye_filenames
    feature_path = os.path.join(dst_dir, experiment_name)
    computed_features.to_csv(feature_path)


# --- Run app ---

if __name__ == "__main__":
    typer.run(main) 


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

        if (img is not None) & (img.mode == 'RGBA'):
                images.append(img)
            
    return filenames, images