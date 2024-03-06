#python imports
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import rasterio
import geopandas as gpd
from rasterio.features import geometry_mask
from rasterio import Affine
from rasterio import open as rasterio_open

#pytorch imports
import torch
from torch import nn
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as TF
import torch.optim as optim
from torch.optim import lr_scheduler

#parameters
batch_size = 5
num_workers = 1
num_classes = 2       #landfill or background
epochs = 100
lr = 1e-3
w_decay = 1e-5
momentum = 0.9
step_size = 10
gamma = 0.5
image_size = 512
patch_width = 512
patch_height= 512

class CustomLandfillDataset(torch.utils.data.Dataset):
  def __init__(self, data, dsType, labelCSV, imgpath, jsonpath, transforms=None, num_classes=batch_size):       #transforms
    self.data = data
    self.labelCSV = labelCSV
    self.json_frame = pd.read_csv(self.labelCSV, usecols=["json index"])
    self.isLandfillList = pd.read_csv(self.labelCSV, usecols=["IsLandfill"]).values.tolist()
    self.num_classes = num_classes
    self.dsType = dsType
    self.imgpath = imgpath
    self.jsonpath = jsonpath

  def transform_train(self, image, mask):
    #convert image and mask to PIL Images
    mask = TF.to_pil_image(mask)
    image = Image.fromarray(image, "RGB")
    #npimg = np.asarray(image)

    #convert PIL image and mask to tensor before returning
    image = TF.to_tensor(image)
    mask = TF.to_tensor(mask)

    #normalise the image as per mean and std dev
    #for ImageNet pre-trained
    image = TF.normalize(image, mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    #for solaris pre-trained
    #image = TF.normalize(image, mean=[0.006479, 0.009328, 0.01123],
    #                            std=[0.004986, 0.004964, 0.004950])

    return image, mask

  def transform_val(self, image, mask):
    image = TF.to_tensor(image)
    #mask = TF.to_tensor(mask)

    #normalise
    image = TF.normalize(image, mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    return image, mask

  def footprint_mask(self, df_path, reference_im, shape=(900, 900), burn_value=255):
    if reference_im:
      if isinstance(reference_im, str):
        reference_im = rasterio_open(reference_im, 'r')

      affine_obj = reference_im.transform
    else:
      affine_obj = None

    df = gpd.read_file(df_path)

    # Create a geospatial raster mask using rasterio's geometry_mask
    mask = geometry_mask(df['geometry'], out_shape=shape, transform=affine_obj, invert=True)

    # Convert the boolean mask to a PyTorch tensor
    mask_tensor = torch.tensor(mask, dtype=torch.uint8)

    # Apply burn_value to object pixels
    mask_tensor[mask_tensor == 1] = burn_value

    return mask_tensor

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    #image id
    img_id = self.data[index][0]
    #path for image
    img_name = self.data[index][1]
    #read the image to extract number of channels
    raster_img = rasterio.open(os.path.join(self.imgpath, img_name)).read()
    #change datatype of image from uint16 to int32
    raster_img = raster_img.astype('uint8')
    #print(raster_img.shape)
    raster_channels = raster_img.shape[0]
    #mask
    json_list = self.json_frame.values.tolist()
    json_name = json_list[img_id-1][0]
    IsLandfill = self.isLandfillList[img_id-1][0]
    if(IsLandfill):
      # fp_mask = sol.vector.mask.footprint_mask(df=os.path.join(self.jsonpath, json_name),
                                              #  reference_im=os.path.join(self.imgpath, img_name))
      fp_mask = self.footprint_mask(df_path=os.path.join(self.jsonpath, json_name), shape=(512, 512), burn_value=255, reference_im=os.path.join(self.imgpath, img_name))

      # fp_mask = fp_mask.astype('uint8')
      fp_mask = fp_mask.to(dtype=torch.uint8)
    else:
      fp_mask = np.zeros((patch_height, patch_width), dtype=np.uint8)

    rgb_image = raster_img.transpose(1,2,0)
    if(raster_channels == 8):
      """
      Worldview-3 spectral bands
      0 - coastal blue, 1 - blue, 2 - green, 3 - yellow
      4 - red, 5 - red edge, 6 - NIR1, 7 - NIR 2
      """
      rgb_image = np.dstack((raster_img[4,:,:],
                             raster_img[2,:,:],
                             raster_img[1,:,:]))
    elif(raster_channels == 4):
      """
      Geoeye-1 sepctral bands
      0 - blue, 1 - green, 2 - red, 3 - NIR
      """
      rgb_image = np.dstack((raster_img[2,:,:],
                             raster_img[1,:,:],
                             raster_img[0,:,:]))


    min = np.min(rgb_image)
    max = np.max(rgb_image)
    #before normalisation, the image should be brought into a 0 - 1 range (standardisation)
    #rgb_image = ((rgb_image - min) / (max - min))     #.astype('uint8')
    rgb_image = rgb_image.astype('uint8')
    #print("raster image-before:",rgb_image)

    #print(image.shape, fp_mask.shape)
    if(self.dsType == 'train'):
      rgb_image, fp_mask = self.transform_train(rgb_image, fp_mask)
    if(self.dsType == 'val'):
      rgb_image, fp_mask = self.transform_val(rgb_image, fp_mask)

    rgb_image_resized = rgb_image.detach().numpy()
    #print("raster image-after:",rgb_image)

    #in case the image is smaller than 512x512
    if((rgb_image.shape[1] < patch_height) or (rgb_image.shape[2] < patch_width)):
      rgb_image_resized = np.zeros((patch_height, patch_width), dtype=np.uint8)
      rgb_image_resized[0:rgb_image.shape[1], 0:rgb_image.shape[2]] = rgb_image[0,:,:]
      rgb_image_resized = np.expand_dims(rgb_image_resized, axis=0)
      for c in range(rgb_image.shape[0]-1):
        resized_raster = np.zeros((patch_height, patch_width), dtype=np.uint8)
        resized_raster[0:rgb_image.shape[1], 0:rgb_image.shape[2]] = rgb_image[c+1,:,:]
        resized_raster = np.expand_dims(resized_raster,axis=0)
        rgb_image_resized = np.vstack((rgb_image_resized, resized_raster))

    #rgb_image_resized = rgb_image_resized.astype('uint8')
    #create another mask variable

    mask = fp_mask
    mask = mask.detach().numpy().squeeze()

    #in case the mask size is smaller than 512x512
    if((fp_mask.shape[1] < patch_height) or (fp_mask.shape[0] < patch_width)):
      mask = np.zeros((patch_height, patch_width), dtype=np.int8)
      mask[0:fp_mask.shape[1], 0:fp_mask.shape[2]] = fp_mask
    #one hot encoding of the mask depending on the number of classes
    mask_hotEnc = torch.zeros(self.num_classes, patch_height, patch_width)
    for n in range(self.num_classes):
      mask_hotEnc[n][mask==n] = 1

    #print("resized image:",rgb_image_resized)
    rgb_image_resized = rgb_image_resized.astype('float32')
    mask = mask.astype('float32')
    mask_hotEnc = mask_hotEnc.detach().numpy().astype('float32')

    img_info = {}
    img_info["RGBimage"] = rgb_image_resized
    img_info["mask"] = mask
    img_info['binary_mask'] = self.footprint_mask(df_path=os.path.join(self.jsonpath, json_name), shape=(512, 512), burn_value=255, reference_im=os.path.join(self.imgpath, img_name))
    img_info["maskHotEnc"] = mask_hotEnc
    img_info["channels"] = raster_channels
    img_info["name"] = img_name
    img_info["image_id"] = img_id

    return img_info