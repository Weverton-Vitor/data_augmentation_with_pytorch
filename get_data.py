import requests
import py7zr
import os

FILE_NAME = 'Pansharpened.7z'
FILES_PATH = FILE_NAME.split('.')[0]

# Download file
if not FILE_NAME in os.listdir():
    # Get file in dropbox
    response = requests.get('https://www.dropbox.com/s/8huwne6of80v5l3/Pansharpened.7z?dl=1')

    print("Downloading files...")
    if response.ok:
        with open(FILE_NAME, 'wb') as file: # Save file
            file.write(response.content)
    print("File downloaded successfully.")

# Extract files
if FILES_PATH not in os.listdir():
    print("Extracting files...")
    with py7zr.SevenZipFile(FILE_NAME, mode='r') as z:
        z.extractall('./')

    # Extract sub folders
    for file_name in os.listdir(FILE_NAME.split('.')[0]):
        if '.7z' in  file_name:
            with py7zr.SevenZipFile(f'./{FILES_PATH}/{file_name}', mode='r') as z:
                z.extractall('./')

    print("Files extracted successfully.")


