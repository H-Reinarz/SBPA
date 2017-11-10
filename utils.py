# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 23:19:31 2017

@author: Jannik
"""
import nunmpy as np

def RemoveChannels(image, channelsToRemove):
    return np.delete(image, np.array(channelsToRemove),2)

def KeepChannels(image, channelsToKeep):
    channels = list(range(image.shape[2]))
    channelsToRemove = [c for c in channels if c not in channelsToKeep]
    return np.delete(image, np.array(channelsToRemove),2)

def NormalizeImage(image):
    for ix in range(image.shape[2]):
        image[:,:,ix] += np.min(image[:,:,ix])
        image[:,:,ix] /= np.max(image[:,:,ix])
    return image

def MergeCHannels(listOfChannels):
    zipped = np.dstack(listOfChannels)
    return zipped

def AddValue(image, value):
    for i in range(image.shape[2]):
        image[:,:,i] + value
    return image