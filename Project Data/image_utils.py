#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!-*- coding:utf-8 -*-
"""image processing utils for pytorch"""
from imageio import imread
import numpy as np
import torchvision.utils as tvls
import torchvision.transforms as transforms

def get_image_from_path(path):
    """
    Read a single image from the input path and return a PIL Image object
    """
    img = imread(path)
    return img

def image_to_tensor(image):
    """
    Convert the input PIL Image object to pytorch Tensor object, channelximage_heightximage_width format
    """
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return trans(image)

def get_tensor_image(path):
    img = get_image_from_path(path)
    return image_to_tensor(img)

def save_tensor_images(images,filename,nrow=None,normalize=True):
    """
    Save the input image tensor array as a picture, the default nrow is 8, which means that 8 pictures are displayed in a row
    """
    if not nrow:
        tvls.save_image(images, filename, normalize=normalize)
    else:
        tvls.save_image(images, filename, normalize=normalize,nrow=nrow)

