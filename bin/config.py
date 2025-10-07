# dataset locations
import os

# === Project root
project_root = '/Users/ntin/Documents/DermaML_local/hawkeye-hands-2024-07-29'

# === Output
output_dir = project_root+'results/'

# === Processed images
image_dir = project_root+'/images_processed/'
image_extension = '.png'

# === Metadata
metadata_file = project_root+'/metadata.csv'
metadata_ref_header = 'hand_image_file'
metadata_ref_extension = '.jpeg'

# === Image Segmentation
# segmentation_file = os.join(
#     '/Users/ntin/',
#     'Models/sam2/notebooks/',
#     '2025-02-23_Hand_Segmentations-Corrected-3.pkl'
# )

# === Tabular data
tabular_feature_file = project_root+'/features-2025-02-24/' 
    # FIXME index files specifically from here
tabular_ref_header = 'filename'
tabular_ref_extension = '.png'