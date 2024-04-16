import os
import pandas as pd
from data_augmentation import apply_data_agumentation_to_train, apply_data_agumentation_to_train_overlay
from lib import get_data, create_masks, split_data, get_only_8_channels, delete_images

ULR_FILE = 'https://www.dropbox.com/s/8huwne6of80v5l3/Pansharpened.7z?dl=1'
FILE_NAME = 'Pansharpened.7z'
DATA_PATH = './HR_TIF_Files'
TRAIN_PATH = DATA_PATH + '/train'
LABELS_PATH = './Pansharpened/PanSharpenedData.csv'
PANSHARPENED_JSON = './LandfillCoordPolygons'
VAL_RATIO = 0.2
ONLY_EIGHT_CHANNLES_IMAGES = True

if __name__ == '__main__':
    # Download the data
    get_data(url=ULR_FILE, file_name=FILE_NAME)

    if ONLY_EIGHT_CHANNLES_IMAGES:
        removed = get_only_8_channels(data_path=DATA_PATH,
                            labels_path=LABELS_PATH)
        
        print(removed)
        # Delete all images with less than 8 channels
        delete_images(images_path=removed)

    # Create masks manually
    create_masks(data_path=DATA_PATH,
                 labels_path=LABELS_PATH,
                 annotations_path=PANSHARPENED_JSON)    

    # Split the train and validation data 
    split_data(data_path=DATA_PATH,
               labels_path=LABELS_PATH,
               val_ratio= VAL_RATIO)
    

    # Apply data augmentaion
    apply_data_agumentation_to_train(train_path=TRAIN_PATH)

        
