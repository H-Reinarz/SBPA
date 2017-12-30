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

def CountPixel(g, attr_name, pixel_min=0, invert=False):
    clusterSet = set() # unique cluster set
    for node in g:
        clusterSet.add(g.node[node][attr_name]) # create unique cluster set
    clusterDict = {c: 0 for c in clusterSet} # cluster set to cluster dict
    for key, value in clusterDict.items():
        for n in g:
            if g.node[n][attr_name] == key:
                clusterDict[key] += g.node[n]['pixel_count']
    
    if not invert:
        clusterDict = {k: v for k, v in clusterDict.items() if v >= pixel_min}
    else:
        clusterDict = {k: v for k, v in clusterDict.items() if v < pixel_min}
    return clusterDict

def OneToThreeChannel(image, deepcopy = True):
    if deepcopy:
        image = np.copy(image)
    return MergeChannels([image,image,image])
    
    