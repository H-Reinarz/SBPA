#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 12:22:51 2017

@author: hre070
"""

import numpy as np
from scipy.spatial.distance import euclidean
from matplotlib import pyplot as plt


dists = []

def dft_summary(freq, width=1):
        
    class_array =  np.zeros(shape=freq.shape, dtype=np.int32)
    
    middle = (class_array.shape[0]/2, class_array.shape[1]/2)
    
    for row in range(class_array.shape[0]):
        for col in range(class_array.shape[1]):
            
            distance = euclidean((row+1, col+1), middle)
            #print(distance)
            dists.append(distance)
            
            class_array[row, col] = distance//width
            
            
            
    plt.imshow(class_array)
    
    class_dict = dict()
    
    for x in np.unique(class_array):
        values = freq[class_array == x]
        mean = np.mean(values)
        class_dict[x] = mean
    
    return class_dict


#======================================================================================
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import gaussian
from numpy.fft import *

image = io.imread("/home/hre070/MA/DJI_0095_CLIP.jpg")
im_gray = rgb2gray(image)

start = 50
width = 400


end  = start+width

clip = im_gray[start:end, start:end]


#clip = im_gray

clip = gaussian(clip, sigma=20)


#Apply shifted furier transform
dft = fftshift(fft2(clip))

dft_log = np.log(np.abs(dft))







#in_shape = (10, 20)
#ring_width = 1
#
#dft = np.random.randint(60, 9999, size=in_shape)/10000

result = dft_summary(dft_log, 10)

plot_values = np.array(list(result.values()))

deriv1 = np.diff(plot_values)
deriv2 = np.diff(deriv1)


f, ax = plt.subplots(nrows=2, figsize=(15,15))

ax[0].imshow(clip, cmap="gray")
ax[1].imshow(dft_log)


f, ax = plt.subplots(nrows=3, figsize=(10,10), sharex=True)

ax[0].plot(plot_values)
ax[1].plot(deriv1, color="r")
ax[2].plot(deriv2, color="g")
    
print(result)