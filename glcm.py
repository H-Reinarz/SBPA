# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:09:40 2017

@author: Jannik
"""
import matplotlib.pyplot as plt

import numpy as np

from skimage import io
from skimage.util.shape import view_as_windows
from skimage.util import img_as_ubyte

from mahotas.features import haralick


image = io.imread("D:/janni/Documents/Geographie/Masterarbeit/Data/ra_neu/ra3/ra3_small.jpg", as_grey = True)



def Haralick_Textures(img, window_x, window_y, distance = 1):
    """Returns an un-normalized 13 channel haralick texture image. Input image needs to be 
    grey-scaled and of type uint. Requires mahotas module."""
    out = np.zeros((img.size, 13), dtype = np.double)
    
    img_tmp = np.lib.pad(img, [window_y//2,window_x//2], "symmetric")
    
    window_shape = (window_x, window_y)
    
    windowed_array = view_as_windows(img_tmp, window_shape)
    
    windowed_array_flat = np.reshape(windowed_array, (img.size, window_shape[0], window_shape[1]))
        
    for i, window in enumerate(windowed_array_flat):
        haralick_features = haralick(f = window, return_mean = True, distance = distance)
        out[i] = haralick_features
    
    out = np.reshape(out, (img.shape[0], img.shape[1], 13))
    
    return out

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

image_uint = img_as_ubyte(image)
haral = Glcm(image_uint, 3,3)
final = normalize_image(haral)

f, ax = plt.subplots(ncols = 4, nrows = 4, figsize=(14, 14), sharex=True, sharey=True,
                       subplot_kw={'adjustable': 'box-forced'})
ax[0, 0].imshow(final[:,:,0], cmap='viridis')
ax[0, 1].imshow(final[:,:,1], cmap='viridis')
ax[0, 2].imshow(final[:,:,2], cmap='viridis')
ax[0, 3].imshow(final[:,:,3], cmap='viridis')
ax[1, 0].imshow(final[:,:,4], cmap='viridis')
ax[1, 1].imshow(final[:,:,5], cmap='viridis')
ax[1, 2].imshow(final[:,:,6], cmap='viridis')
ax[1, 3].imshow(final[:,:,7], cmap='viridis')
ax[2, 0].imshow(final[:,:,8], cmap='viridis')
ax[2, 1].imshow(final[:,:,9], cmap='viridis')
ax[2, 2].imshow(final[:,:,10], cmap='viridis')
ax[2, 3].imshow(final[:,:,11], cmap='viridis')
ax[3, 0].imshow(final[:,:,12], cmap='viridis')

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()