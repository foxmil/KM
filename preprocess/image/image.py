# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 09:41:10 2017

@author: miller89
"""
import os
import cv2
import os.path as osp
import numpy as np
from PIL import Image
from scipy.misc import imsave



# Loads a greyscale image from a path.
def load_greyscale_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img[:, :, np.newaxis]



# Loads and image.
def load_img(path, rgb=True):
    if rgb is True:
        img = cv2.imread(path)
    else:
        img = load_greyscale_img(path)
        
    return img

# Crops an image at its center to a specified size.
def center_crop(img, size):
    # Find the center of the image so that a crop can be made around it.
    shape = img.shape
    center = np.array([shape[0] / 2, shape[1] / 2])
    
    # Find how far to go on each side.
    img_del = np.array([size[0] / 2, size[1] / 2])
    
    # Determine the corners to crop the image at.
    ul = (center - img_del).astype(int)
    br = (center + img_del).astype(int)
    
    crop_img = img[ul[0]:br[0], ul[1]:br[1]]

    return crop_img
    


# Generates a possible number of image representations (4 for rectangular, 8 for squares)
def gen_img_reps(img):
    # Ensure the picture is square so that rotations can be properly applied if not then only apply flips
    shape = img.shape
    rot_flag = True if shape[0] == shape[1] else False
    
    reps = []
    reps.append(img)
    reps.append(cv2.flip(img, 0))
    reps.append(cv2.flip(img, 1))
    reps.append(cv2.flip(img, -1))
    
    if (rot_flag == True):
        img90 = np.rot90(img)
        reps.append(cv2.flip(img90, 0))
        reps.append(cv2.flip(img90, 1))
        reps.append(cv2.flip(img90, -1))

    return reps



# Loads all images from a directory with a certain extension.
def load_imgs_from_dir(path, crop=None, rgb=True, ext='tif'):
    files = os.listdir(path)
    
    # Only take image files in the directory.
    img_names = [file for file in files if file.endswith('.' + ext)]
    
    # Load each image and add it to the array.
    imgs = []
    for name in img_names:
        img_path = os.path.join(path, name)
        img = cv2.imread(img_path) if rgb == True else load_greyscale_img(img_path)
        
        # Crop the image if requested.
        img = img if crop == None else center_crop(img, crop)
        
        # Add the image to the imgs array
        imgs.append(img)
        
    return imgs
    


# Creates an RGB image from greyscale images (at least two).
def make_RGB(r_img, g_img, b_img=None):
    # Check that the r and g channels have the same dimensions for concecation.
    shape = r_img.shape 
    if shape != g_img.shape:
        raise Exception
    
    # If the b channel exists then also check that it has the same shot
    if b_img is not None:
        if shape != b_img.shape:
            raise Exception
    
    # Create the RGB image. from the components
    img = np.empty((shape[0], shape[1], 3), dtype=np.uint8)
    img[:, :, 0] = r_img
    img[:, :, 1] = g_img
    
    if b_img is None:
        img[:, :, 2].fill(123)
    else:
        img[:, :, 2] = b_img
    
    return img



# Identifies the number of classes a set of images has. 
# Path should point to a directory with subfolders labeling each class.
def find_num_classes(path, ext='tif'):
    # Find all folders in the working directory.
    folders = [x for x in os.listdir(path) if osp.isdir(osp.join(path, x))]
    
    # For each folder in the find if there are any files with the desired extension.
    count = 0
    for folder in folders:
        folder_path = osp.join(path, folder)

        # If a file was         
        files = [x for x in os.listdir(folder_path) if not osp.isdir(osp.join(path, x))]
        for file in files:
            if file.endswith('.' + ext):
                count += 1
                break    

    return count



# Generates a list of all the images in the path with the classes in the subdirectory
def generate_img_list(path, ext='tif'):
    # Find all folders in the working directory.
    folders = [x for x in os.listdir(path) if osp.isdir(osp.join(path, x))]
    
    # For each folder in the find if there are any files with the desired extension.
    file_list = []
    for folder in folders:
        folder_path = osp.join(path, folder)

        # If a file has the proper extension add it to the list      
        files = [x for x in os.listdir(folder_path) if not osp.isdir(osp.join(path, x))]
        for file in files:
            if file.endswith('.' + ext):
                file_list.append(osp.join(folder_path, file))
    
    return file_list




