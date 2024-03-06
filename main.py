from lib import get_data, create_masks, split_data

ULR_FILE = 'https://www.dropbox.com/s/8huwne6of80v5l3/Pansharpened.7z?dl=1'
FILE_NAME = 'Pansharpened.7z'
TRAIN_PATH = './HR_TIF_Files'
TRAIN_LABELS = './Pansharpened/PanSharpenedData.csv'
PANSHARPENED_JSON = './LandfillCoordPolygons'
TEST_RATIO = 0.2



if __name__ == '__main__':
    get_data(url=ULR_FILE, file_name=FILE_NAME)
    create_masks(train_path=TRAIN_PATH,
                 train_labels=TRAIN_LABELS,
                 annotations_path=PANSHARPENED_JSON)