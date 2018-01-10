# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 23:19:31 2017

@author: Jannik
"""
import numpy as np


def remove_channels(image, channels_to_remove):
    '''Takes list of channel indices to remove from an image'''
    
    return np.delete(image, np.array(channels_to_remove),2)



def keep_channels(image, channels_to_keep):
    '''Takes list of channel indices to keep (all others will be dropped)'''
    
    channels = list(range(image.shape[2]))
    channels_to_remove = [c for c in channels if c not in channels_to_keep]
    return np.delete(image, np.array(channels_to_remove),2)



def normalize_image(image, deepcopy=True):
    '''Normalize Image to 0.0 - 1.0'''
    if deepcopy:
        image = np.copy(image)
    if image.ndim == 2:
        image[:,:] += abs(np.min(image[:,:]))
        image[:,:] /= np.max(image[:,:])
    elif image.ndim > 2:
        for ix in range(image.shape[2]):
            image[:,:,ix] += abs(np.min(image[:,:,ix]))
            image[:,:,ix] /= np.max(image[:,:,ix])
    return image



def merge_channels(list_of_channels):
    '''Merge list of single channel images to multi channel image'''
    
    zipped = np.dstack(list_of_channels)
    return zipped



def add_value(image, value):
    '''Add a value to all pixels'''
    
    image += value
    return image



def value_to_value(image, value_old, value_new):
    '''Set specific pixel to specific value'''
    
    image[image == value_old] = value_new
    return image



def highest_value_minus_one(image, value=1):
    '''Substract 1 from 255 in an image'''
    
    image[image == 255] -= value
    return image



def image_from_array(array, image):
    '''Brings flattened array into the shape of an image'''
    
    return np.reshape(array, (image.shape[0], image.shape[1]))



def one_to_three_channels(image, deepcopy = True):
    '''Makes a single channel image to a multi channel image by duplicating
    channel'''
    
    if deepcopy:
        image = np.copy(image)
    return MergeChannels([image,image,image])
    
    