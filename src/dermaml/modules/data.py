import pandas as pd
import torch
import os
from torch.utils.data import Dataset
from PIL import Image

# -- locally corrupted images --
corrupted_images = ['e372cfe9-dbf4-4308-86ca-fd17547d6b51.jpeg',
                        '748c093e-8c5f-4879-b4aa-8da5d954768f.jpeg',
                        '732c435e-5470-4aa0-b51c-5feffe76863f.jpeg',
                        '056ce0ab-bb2b-4792-aec7-956c754dc852.jpeg']

class HawkeyeHandsDataset(Dataset):
    def __init__(
            self, 
            metadata_file, 
            img_dir, 
            filename_jpeg, 
            label='age',
            metadata_filename_column = 'hand_image_file',
            transform=None, target_transform=None
            ):
        
        # load metadata
        self.metadata_all = pd.read_csv(metadata_file)
        self.metadata_filename_column = metadata_filename_column
        # filter metadata to only include images in dataset_filenames
        self.metadata = self.metadata_all[self.metadata_all[metadata_filename_column].isin(filename_jpeg)]
            
        # set labels
        self.classes = self.metadata[label]

        # load image files
        self.img_dir = img_dir
        self.filename_jpeg = filename_jpeg

        # set transformations
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, idx):
        fname = self.filename_jpeg[idx]
        img_path = os.path.join(self.img_dir, fname)
        image = Image.open(img_path)

        try:
            if fname in corrupted_images:
                raise IndexError(f"Skipping corrupted image file: {fname}, {idx}")
            label = self.metadata.loc[self.metadata[self.metadata_filename_column] == fname, 'age'].values[0]
        except IndexError:
            raise IndexError(f"No label found for image file: {fname}, {idx}")
        
        if self.transform:
            image = self.transform(image)
        
        
        label = torch.tensor(float(label), dtype=torch.float32)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label