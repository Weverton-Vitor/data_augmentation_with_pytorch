from data_augmentation import apply_data_agumentation_to_train, apply_data_agumentation_to_train_overlay
from lib import get_data, create_masks, split_data

ULR_FILE = 'https://www.dropbox.com/s/8huwne6of80v5l3/Pansharpened.7z?dl=1'
FILE_NAME = 'Pansharpened.7z'
DATA_PATH = './HR_TIF_Files'
TRAIN_PATH = DATA_PATH + '/train'
LABELS_PATH = './Pansharpened/PanSharpenedData.csv'
PANSHARPENED_JSON = './LandfillCoordPolygons'
TEST_RATIO = 0.2

if __name__ == '__main__':
    get_data(url=ULR_FILE, file_name=FILE_NAME)
    create_masks(data_path=DATA_PATH,
                 labels_path=LABELS_PATH,
                 annotations_path=PANSHARPENED_JSON)

    split_data(data_path=DATA_PATH,
               labels_path=LABELS_PATH,
               test_ratio= TEST_RATIO)
    
    apply_data_agumentation_to_train(train_path=TRAIN_PATH)
