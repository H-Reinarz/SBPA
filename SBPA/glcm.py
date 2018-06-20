# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:09:40 2017

@author: Jannik
"""
import numpy as np
from skimage.util.shape import view_as_windows
from mahotas.features import haralick

def Haralick_Textures(img, window_x, window_y, distance = 1):
    """Returns an un-normalized 13 channel haralick texture image. Input image needs to be 
    grey-scaled and of type uint. Window Dimensions need to be odd. Requires mahotas module."""
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
