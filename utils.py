# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 23:19:31 2017

@author: Jannik
"""
import numpy as np

def RemoveChannels(image, channelsToRemove):
    return np.delete(image, np.array(channelsToRemove),2)

def KeepChannels(image, channelsToKeep):
    channels = list(range(image.shape[2]))
    channelsToRemove = [c for c in channels if c not in channelsToKeep]
    return np.delete(image, np.array(channelsToRemove),2)

def NormalizeImage(image, deepcopy=True):
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

def MergeChannels(listOfChannels):
    zipped = np.dstack(listOfChannels)
    return zipped

def AddValue(image, value):
    image += value
    return image

def ZerosToOne(image, value):
    image[image == 0] += value
    return image

def HighestValueMinusOne(image, value=1):
    image[image == 255] -= value
    return image

def ImageFromArray(array, image):
    return np.reshape(array, (image.shape[0], image.shape[1]))