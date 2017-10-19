# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:09:40 2017

@author: Jannik
"""

import matplotlib.pyplot as plt
import numpy as np

from skimage.feature import greycomatrix, greycoprops
from skimage import data
from skimage import io
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte

from collections import namedtuple

image = data.camera()
#image = io.imread("D:/janni/Documents/Geographie/Masterarbeit/Data/ResearchArea/RA1/orthoRA.jpg")
#image = rgb2gray(image)
#image = img_as_ubyte(image)


def Glcm(img, window_size, distance, angle):
    
    measure_list = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]
    propTuple = namedtuple('GLCM_Props', measure_list)
    d = dict.fromkeys(measure_list)
    
    if window_size % 2 == 0:
        raise ValueError("Window Size must be odd")
        
    padding = int(window_size/2)
    img = np.lib.pad(img, [padding,padding], "symmetric")
    
    
    for m in measure_list:
        out = np.zeros_like(img)
        for y in range(padding,img.shape[0]):
            for x in range(padding,img.shape[1]):
                patch = img[y-padding:y+padding+1,
                            x-padding:x+padding+1]
                glcm = greycomatrix(patch, distance, angle, 256, symmetric=True, normed=True)
                #glcm = greycomatrix(patch, distance, **glcm_kwargs)
                out[y,x] = greycoprops(glcm, m)[0, 0]
                
        d[m] = out[padding:-padding,padding:-padding]
    
    return propTuple(**d)

bob = Glcm(image, 21, [5], [0])
