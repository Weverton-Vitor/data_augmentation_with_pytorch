import requests
import py7zr
import os
import shutil

from CustomLandfillDataset import CustomLandfillDataset
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd


def get_data(url: str, file_name: str):
    files_path = file_name.split('.')[0]

    # Download file
    if not file_name in os.listdir():
        # Get file in dropbox
        response = requests.get(url=url)

        print("-> Downloading files...")
        if response.ok:
            with open(file_name, 'wb') as file: # Save file
                file.write(response.content)
            print("-> File downloaded successfully.")
        else:
            print(f"-> Error: response isn't ok : {response.status_code}")
    
    else:
        print(f"-> File {file_name} is already downloaded, to donwload again remove it manually")

    # Extract files
    if files_path not in os.listdir():
        print("-> Extracting files...")
        with py7zr.SevenZipFile(file_name, mode='r') as z:
            z.extractall('./')

        # Extract sub folders
        for file_name in os.listdir(file_name.split('.')[0]):
            if '.7z' in  file_name:
                with py7zr.SevenZipFile(f'./{files_path}/{file_name}', mode='r') as z:
                    z.extractall('./')

        print("-> Files extracted successfully.")

    else:
        print(f"-> File {file_name} is already extracted, to extract again remove it and its subfolders manually")


def create_masks(data_path: str, labels_path: str, annotations_path):
    files = [f for f in os.listdir(data_path) if os.path.isfile(f'{data_path}/{f}')]
    if not files:
        print('-> Data path empty or already preprocessed')
    else:
        dataFrame = pd.read_csv(labels_path, usecols=["Idx", "Image Index", "IsLandfill"])
        train_dataset = CustomLandfillDataset(data=dataFrame.values.tolist(),
                                            dsType = 'train',
                                            transforms=None,
                                            labelCSV=labels_path,
                                            imgpath=data_path,
                                            jsonpath=annotations_path)

        if 'masks_temp' not in os.listdir(data_path):
            print(f"-> Create folder masks_temp in {f'{data_path}/masks_temp'}...")
            os.mkdir(f'{data_path}/masks_temp')

            print("-> Create masks...")
            for img in train_dataset:
                mask = Image.fromarray(img['binary_mask'].numpy(), "L")
                mask.save(f'{data_path}/masks_temp/{img["name"].split(".")[0]}_mask.png')

            print('-> Masks created successfully')

        else:
            print(f"-> 'masks_temp' folder already exist inside {data_path} folder")


def split_data(data_path: str, labels_path: str, test_ratio: float,) ->  None:
    df = pd.read_csv(labels_path, usecols=["Idx", "Image Index", "IsLandfill"])
    label = pd.read_csv(labels_path, usecols=["IsLandfill"])

    train_df, test_df, train_lab, test_lab = train_test_split(df,
                                                                label,
                                                                test_size = test_ratio,
                                                                shuffle=True, 
                                                                stratify=label)
    
    # if 'train' not in os.listdir(data_path):
    #     print(f"-> Create folder train in {f'{data_path}/train'}...")
    #     print(f"-> Create folder masks in {f'{data_path}/train/masks'}...")

    #     os.mkdir(f'{data_path}/train')
    #     os.mkdir(f'{data_path}/train/masks')
        
    #     print("Move images and masks to train folder")
    #     for image in train_df['Image Index']:
    #         shutil.move(f'{data_path}/{image}', f'{data_path}/train/{image}')
    #         shutil.move(f'{data_path}/masks_temp/{image.split(".")[0]}_mask.png', f'{data_path}/train/masks/{image.split(".")[0]}_mask.png')

    # Create folder train and move the selected data
    def split(path: str, df: pd.DataFrame) ->  None:
        if path not in os.listdir(data_path):
            print(f"-> Create folder train in {f'{data_path}/train'}...")
            print(f"-> Create folder masks in {f'{data_path}/train/masks'}...")

            os.mkdir(f'{data_path}/{path}')
            os.mkdir(f'{data_path}/{path}/masks')
            
            print(f"-> Move images and masks to {path} folder")
            for image in df['Image Index']:
                shutil.move(f'{data_path}/{image}', f'{data_path}/{path}/{image}')
                shutil.move(f'{data_path}/masks_temp/{image.split(".")[0]}_mask.png', f'{data_path}/{path}/masks/{image.split(".")[0]}_mask.png')

    if 'train' not in os.listdir(data_path):
        split(path='train', df=train_df)
    
    if 'test' not in os.listdir(data_path):
        split(path='test', df=test_df)
    
    print(f"-> Deleting {data_path}/masks_temp/'")
    if 'masks_temp' in os.listdir(data_path):
        shutil.rmtree(f'{data_path}/masks_temp/')
