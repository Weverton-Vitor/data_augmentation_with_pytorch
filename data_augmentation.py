import os
import tifffile
from PIL import Image
import numpy as np
from torchvision.transforms import functional as TF
from skimage.transform import rotate


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

        # print('----------------------------------------')
        # print(image.min(), image.max(), image.dtype)
        image = TF.to_tensor(image.astype(np.int64))
        # print(image.numpy().min(), image.numpy().max(), image.dtype)
        
        #random horizontal flip
        image_h_flip = TF.hflip(image)
        mask_h_flip = TF.hflip(mask)
        # print(image_h_flip.numpy().min(), image_h_flip.numpy().max(), image_h_flip.numpy().dtype)

        with tifffile.TiffWriter(f"{train_path}/{image_name}_h_flip.tif") as tif:
            tif.write(image_h_flip.numpy(), shape=image.shape, photometric="separated")
        mask_h_flip.save(f"{train_path}/masks/{mask_name}_h_flip_mask.png")

        #random vertical flip
        image_v_flip = TF.vflip(image)
        mask_v_flip = TF.vflip(mask)
        # print(image_v_flip.numpy().min(), image_v_flip.numpy().max(), image_v_flip.numpy().dtype)
        with tifffile.TiffWriter(f"{train_path}/{image_name}_v_flip.tif") as tif:
            tif.write(image_v_flip.numpy(), shape=image.shape, photometric="separated")
        mask_v_flip.save(f"{train_path}/masks/{mask_name}_v_flip_mask.png")


        # # random rotation
        rotated_image = np.zeros_like(image)
        for channel in range(image.shape[0]):
            rotated_image[channel, :, :] = rotate(image[channel, :, :], angle=25, preserve_range=True)

        mask_rotated = TF.rotate(mask, angle=25.0)
        # print(rotated_image.numpy().min(), rotated_image.numpy().max(), rotated_image.numpy().dtype)

        with tifffile.TiffWriter(f"{train_path}/{image_name}_rotated.tif") as tif:
            tif.write(rotated_image, shape=image.shape, photometric="separated")
        mask_rotated.save(f"{train_path}/masks/{mask_name}_rotated_mask.png")

        # print('----------------------------------------')

    print("-> Finish data augmentation")
