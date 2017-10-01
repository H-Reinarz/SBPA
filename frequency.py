#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 12:22:51 2017

@author: hre070
"""

import numpy as np
from scipy.spatial.distance import euclidean
from matplotlib import pyplot as plt


def create_radius_map(shape, array_dtype=np.float32):
    
    radius_array =  np.zeros(shape=shape, dtype=array_dtype)
    
    middle = (radius_array.shape[0]/2, radius_array.shape[1]/2)
    
    for row in range(radius_array.shape[0]):
        for col in range(radius_array.shape[1]):
            
            distance = euclidean((row+1, col+1), middle)
            #print(distance)         
            radius_array[row, col] = int(distance)

    return radius_array



def dft_summary(freq, width=1):
            
    class_array = create_radius_map(freq.shape)//width
            
    plt.imshow(class_array)
    
    class_dict = dict()
    
    for x in np.unique(class_array):
        values = freq[class_array == x]
        mean = np.mean(values)
        class_dict[x] = mean
    
    return class_dict


def xcut_auc(value_dict, cut_perc, width):
    
    area = sum(value_dict.values())
    cut_value = area/100*cut_perc

    for key, value in value_dict.items():
        if cut_value < value:            
          return key - int(round(cut_value/(value/width)))
        else:
            cut_value -= value
    

def lowpass_filter(shifted_dft, cut, stretch, upper, lower):
    
    span = upper - lower
    sigmoid = [upper-(3*(x**2)-2*(x**3))*span for x in list(np.linspace(0,1, stretch))]

    radius = create_radius_map(shifted_dft.shape)
    
    #Filter creation
    filter_array = np.zeros(shape=radius.shape, dtype=np.float32)
    filter_array[radius <= cut] = upper
    filter_array[cut < radius] = lower

    for ix, f in enumerate(sigmoid):
        add = ix+1 
        filter_array[radius == cut+add] = f
    
    
    return shifted_dft*filter_array
    



class frequency_setup:
    
    def __init__(self, image):
        self.dft = fftshift(fft2(image))
            
    def dft_log(self):
        return np.log(np.abs(self.dft))
    
    def summarize(self, width):
        return dft_summary(self.dft, width)
    
    def high_cut_value(self, percentage):
        pass

#======================================================================================

if __name__ == "__main__":
        
    
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
    
    
    #Generate random noise within brightness spectrum of the image
    
    factor = 10000
    lower = clip.min()*factor
    upper = clip.max()*factor
    
    noise = np.random.randint(lower, upper, size=clip.shape)/factor
    
    
    clip += noise
    
    
    
    #clip = im_gray
    
    #
    
    #Apply shifted furier transform
    dft = fftshift(fft2(clip))
    
    dft_log = np.log(np.abs(dft))
    
    
    def butter2d_lp(shape, f, n, pxd=1):
        pxd = float(pxd)
        rows, cols = shape
        x = np.linspace(-0.5, 0.5, cols)  * cols / pxd
        y = np.linspace(-0.5, 0.5, rows)  * rows / pxd
        radius = np.sqrt((x**2)[np.newaxis] + (y**2)[:, np.newaxis])
        filt = 1 / (1.0 + (radius / f)**(2*n))
        return filt
    
    filt = butter2d_lp(clip.shape, 2, 5, pxd=50)
    dft_new = dft * filt
    new_image = np.abs(np.fft.ifft2(np.fft.ifftshift(dft_new)))  
    
    plt.imshow(filt)
    
    #in_shape = (10, 20)
    #ring_width = 1
    #
    #dft = np.random.randint(60, 9999, size=in_shape)/10000
    
    result = dft_summary(dft_log, 5)
    
    x_cut = xcut_auc(result, 80, 5)
    
    plot_values = np.array(list(result.values()))
    
    deriv1 = np.diff(plot_values)
    deriv2 = np.diff(deriv1)
    
    
    f, ax = plt.subplots(nrows=5, figsize=(20,20))
    
    ax[0].imshow(clip, cmap="gray")
    ax[1].imshow(dft_log)
    ax[2].imshow(filt)
    ax[3].imshow(np.log(np.abs(dft_new)))
    ax[4].imshow(new_image, cmap="gray")    
    
#    f, ax = plt.subplots(nrows=3, figsize=(10,10), sharex=True)
#    
#    ax[0].plot(plot_values)
#    ax[0].axvline(x_cut, color='r')
#    ax[1].plot(deriv1, color="r")
#    ax[2].plot(deriv2, color="g")
        
    #print(result)