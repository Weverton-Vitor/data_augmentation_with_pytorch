import os
import random
import tifffile
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import functional as TF
from torchvision.transforms import v2
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
            rotated_image[channel, :, :] = rotate(image.numpy()[channel, :, :], angle=25, preserve_range=True)

        mask_rotated = TF.rotate(mask, angle=25.0)
        # print(rotated_image.numpy().min(), rotated_image.numpy().max(), rotated_image.numpy().dtype)

        with tifffile.TiffWriter(f"{train_path}/{image_name}_rotated.tif") as tif:
            tif.write(rotated_image, shape=image.shape, photometric="separated")
        mask_rotated.save(f"{train_path}/masks/{mask_name}_rotated_mask.png")

        # elastic transform
        elastic_transformer = v2.ElasticTransform(alpha=250.0)
        image_elastic = elastic_transformer(image)
        mask_elastic = elastic_transformer(mask)
        # print(image_elastic.numpy().min(), image_elastic.numpy().max(), image_elastic.numpy().dtype)
        with tifffile.TiffWriter(f"{train_path}/{image_name}_elastic.tif") as tif:
            tif.write(image_elastic.numpy(), shape=image.shape, photometric="separated")
        mask_elastic.save(f"{train_path}/masks/{mask_name}_elastic_mask.png")

        # crop with random size

        # Generate random values for top and left coordinates
        crop_size = random.randint(1, image.shape[1]-100)
        top = random.randint(0, image.shape[1] - crop_size)
        left = random.randint(0, image.shape[1] - crop_size)

        # The size of the cropped region is the same as crop_size
        size = crop_size

        # image_crop_resized = TF.resized_crop(image, height=crop_size, width=crop_size, top=top, left=left, size=(512, 512))
        # mask_crop_resized = TF.resized_crop(mask, height=crop_size, width=crop_size, top=top, left=left, size=size)
        # # print(image_crop_resized.numpy().min(), image_crop_resized.numpy().max(), image_crop_resized.numpy().dtype)
        # with tifffile.TiffWriter(f"{train_path}/{image_name}_crop_resized.tif") as tif:
        #     tif.write(image_crop_resized.numpy(), shape=image.shape, photometric="separated")
        # mask_crop_resized.save(f"{train_path}/masks/{mask_name}_crop_resized_mask.png")



        # print('----------------------------------------')

    print("-> Finish data augmentation")


def apply_data_agumentation_to_train_overlay(train_path):
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
        mask = TF.to_tensor(mask)
        # print('init', mask.shape, image.shape)
        # print(image.numpy().min(), image.numpy().max(), image.dtype)
        
        for i in range(3):
            #random horizontal flip
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            # print(i, 'nomarl', mask.shape, image.shape)
            # print(image.numpy().min(), image.numpy().max(), image.numpy().dtype)

            if random.random() > 0.5:
                #random vertical flip
                image = TF.vflip(image)
                mask = TF.vflip(mask)
                # print(i, 'flip', mask.shape, image.shape)
                # print(image.numpy().min(), image.numpy().max(), image.numpy().dtype)

            if random.random() > 0.5:
                # # random rotation
                rotated_image = np.zeros_like(image)
                for channel in range(image.shape[0]):
                    rotated_image[channel, :, :] = rotate(image.numpy()[channel, :, :], angle=25, preserve_range=True)

                image = torch.from_numpy(rotated_image)
                mask = TF.rotate(mask, angle=25.0)
                # print(i, 'rotate', mask.shape, image.shape)
                # print(rotated_image.numpy().min(), rotated_image.numpy().max(), rotated_image.numpy().dtype)

            if random.random() > 0.5:
                # elastic transform
                elastic_transformer = v2.ElasticTransform(alpha=250.0)
                image = elastic_transformer(image)
                # print('l: ', mask.shape, image.shape)
                mask = elastic_transformer(mask)
                # print(i, 'elastic', mask.shape, image.shape)
                # print(image.numpy().min(), image.numpy().max(), image.numpy().dtype)

            if random.random() > 0.5:

                # crop with random size
                # Generate random values for top and left coordinates
                crop_size = random.randint(1, image.shape[1]-(image.shape[1]*0.25))
                top = random.randint(0, image.shape[1] - crop_size)
                left = random.randint(0, image.shape[1] - crop_size)

                # The size of the cropped region is the same as crop_size
                size = crop_size

                image = TF.resized_crop(image, height=crop_size, width=crop_size, top=top, left=left, size=(512, 512))
                mask = TF.resized_crop(mask, height=crop_size, width=crop_size, top=top, left=left, size=(512, 512))
                # print(image.numpy().min(), image.numpy().max(), image.numpy().dtype)
                # print(i, 'crop', mask.shape, image.shape)

            with tifffile.TiffWriter(f"{train_path}/{image_name}_augmented_{i}.tif") as tif:
                tif.write(image.numpy(), shape=image.shape, photometric="separated")

            TF.to_pil_image(mask).save(f"{train_path}/masks/{mask_name}_augmented_{i}_mask.png")

    print("-> Finish data augmentation")