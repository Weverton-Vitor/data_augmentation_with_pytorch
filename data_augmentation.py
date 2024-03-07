import os
import tifffile
from PIL import Image
import numpy as np
from torchvision.transforms import functional as TF


def apply_data_agumentation_to_train(train_path):
    images = [f'{train_path}/{img}' for img in os.listdir(train_path) if os.path.isfile(f'{train_path}/{img}')]
    masks = [f'{train_path}/masks/{mask}' for mask in os.listdir(train_path+'/masks') if os.path.isfile(f'{train_path}/masks/{mask}')]

    print("-> Applying vertical flip, horizontal flip and rotate separately")
    for img_path in images:
        image = tifffile.imread(img_path)
        image_name = img_path.split('/')[-1].split('.')[0]

        mask = list(filter(lambda x: image_name in x, masks))[0]
        mask_name = mask.split('.')[1].split('/')[-1].replace('_mask', '')
        mask = Image.open(mask)

        image = TF.to_tensor(image.astype(np.float32))
        
        #random horizontal flip
        image_h_flip = TF.hflip(image)
        mask_h_flip = TF.hflip(mask)
        with tifffile.TiffWriter(f"{train_path}/{image_name}_h_flip.tif") as tif:
            tif.write(image_h_flip.numpy(), shape=image.shape)
        mask_h_flip.save(f"{train_path}/masks/{mask_name}_h_flip_mask.png")

        #random vertical flip
        image_v_flip = TF.vflip(image)
        mask_v_flip = TF.vflip(mask)
        with tifffile.TiffWriter(f"{train_path}/{image_name}_v_flip.tif") as tif:
            tif.write(image_v_flip.numpy(), shape=image.shape)
        mask_v_flip.save(f"{train_path}/masks/{mask_name}_v_flip_mask.png")


        # # random rotation
        image_rotated = TF.rotate(image, angle=10.0)
        mask_rotated = TF.rotate(mask, angle=10.0)
        with tifffile.TiffWriter(f"{train_path}/{image_name}_rotated.tif") as tif:
            tif.write(image_rotated.numpy(), shape=image.shape)
        mask_rotated.save(f"{train_path}/masks/{mask_name}_rotated_mask.png")


    print("-> Finish data augmentation")
