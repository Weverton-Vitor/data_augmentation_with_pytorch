import requests
import py7zr
import os
from CustomLandfillDataset import CustomLandfillDataset
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
import numpy as np
import os


def get_data(url: str, file_name: str):
    files_path = file_name.split('.')[0]

    # Download file
    if not file_name in os.listdir():
        # Get file in dropbox
        response = requests.get(url=url)

        print("Downloading files...")
        if response.ok:
            with open(file_name, 'wb') as file: # Save file
                file.write(response.content)
            print("File downloaded successfully.")
        else:
            print(f"Error: response isn't ok : {response.status_code}")

    # Extract files
    if files_path not in os.listdir():
        print("Extracting files...")
        with py7zr.SevenZipFile(file_name, mode='r') as z:
            z.extractall('./')

        # Extract sub folders
        for file_name in os.listdir(file_name.split('.')[0]):
            if '.7z' in  file_name:
                with py7zr.SevenZipFile(f'./{files_path}/{file_name}', mode='r') as z:
                    z.extractall('./')

        print("Files extracted successfully.")


def create_masks(train_path: str, train_labels: str, annotations_path):
    dataFrame = pd.read_csv(train_labels, usecols=["Idx", "Image Index", "IsLandfill"])
    label = pd.read_csv(train_labels, usecols=["IsLandfill"])

    train_dataset = CustomLandfillDataset(data=dataFrame.values.tolist(),
                                        dsType = 'train',
                                        transforms=None,
                                        labelCSV=train_labels,
                                        imgpath=train_path,
                                        jsonpath=annotations_path)

    if 'masks' not in os.listdir(train_path):
        print(f"Create folder masks in {f'{train_path}/masks'}...")
        os.mkdir(f'{train_path}/masks')

        print("Create masks...")
        for img in train_dataset:
            mask = Image.fromarray(img['binary_mask'].numpy(), "L")
            mask.save(f'{train_path}/masks/{img["name"].split(".")[0]}_mask.png')

        print('Masks created successfully')

    else:
        print(f"'mask' folder already exist inside {train_path} folder")


def split_data(train_rate: float, train_labels: str) -> None:
    dataFrame = pd.read_csv(train_labels, usecols=["Idx", "Image Index", "IsLandfill"])
