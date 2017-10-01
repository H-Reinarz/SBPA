#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 12:22:51 2017

@author: hre070
"""

import numpy as np
from scipy.spatial.distance import euclidean
from matplotlib import pyplot as plt
from math import pi
from numpy import fft

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
    

def lowpass_filter(shifted_dft, radius, stretch, upper, lower):
    
    span = upper - lower
    sigmoid = [upper-(3*(x**2)-2*(x**3))*span for x in list(np.linspace(0,1, stretch))]

    radius_array = create_radius_map(shifted_dft.shape)
    
    #Filter creation
    filter_array = np.zeros(shape=radius_array.shape, dtype=np.float32)
    filter_array[radius_array <= radius] = upper
    filter_array[radius < radius_array] = lower

    for ix, f in enumerate(sigmoid):
        add = ix+1 
        filter_array[radius_array == radius+add] = f
    
    
    return shifted_dft*filter_array
    

def calc_lbp_radius(dft_shape, radius):
    
    denom = max(dft_shape)
    
    wavelength = (2*pi)/(radius*2*pi/denom)
    
    return int(round(wavelength/2))


class frequency_setup:
    
    def __init__(self, image, width, cut, stretch, upper, lower):
        self.dft = fft.fftshift(fft.fft2(image))
        
        self.summary = dft_summary(self.dft_log(), width)
        
        self.cut_value = xcut_auc(self.summary, cut, width)
        
        self.filtered_dft = lowpass_filter(self.dft, self.cut_value, stretch, upper, lower)
        
        self.lbp_radius = calc_lbp_radius(self.dft.shape, self.cut_value)
            
    def dft_log(self):
        return np.log(np.abs(self.dft))

    def filtererd_dft_log(self):   
        return np.log(np.abs(self.filtered_dft))

#======================================================================================

if __name__ == "__main__":
        
    
    from skimage import io
    from skimage.color import rgb2gray
    from skimage.filters import gaussian
#    from numpy.fft import *
    
    image = io.imread("/home/hre070/MA/DJI_0095_CLIP.jpg")
    im_gray = rgb2gray(image)
    
    start = 50
    width = 400
    
    
    end  = start+width
    
    clip = im_gray[start:end, start:end]
    
    
    #Generate random noise within brightness spectrum of the image
    
#    factor = 10000
#    lower = clip.min()*factor
#    upper = clip.max()*factor
#    
#    noise = np.random.randint(lower, upper, size=clip.shape)/factor
#    
#    
#    clip += noise
    
    
    
    #clip = im_gray
    
    #
    
    #Apply shifted furier transform
#    dft = fftshift(fft2(clip))
    
    dft_log = np.log(np.abs(dft))
    
    
    
    
    result = dft_summary(dft_log, 5)
    
    x_cut = xcut_auc(result, 80, 5)
    
    plot_values = np.array(list(result.values()))
    
    deriv1 = np.diff(plot_values)
    deriv2 = np.diff(deriv1)
    


    #CLASS TEST
    test = frequency_setup(clip, 5, 80, 10, 1.0, 0.1)
    
    print(test.cut_value, test.lbp_radius)


  
#    f, ax = plt.subplots(nrows=2, figsize=(20,20))
#    
#    ax[0].imshow(clip, cmap="gray")
#    ax[1].imshow(dft_log)
#    ax[2].imshow(new_image, cmap="gray")    
    
#    f, ax = plt.subplots(nrows=3, figsize=(10,10), sharex=True)
#    
#    ax[0].plot(plot_values)
#    ax[0].axvline(x_cut, color='r')
#    ax[1].plot(deriv1, color="r")
#    ax[2].plot(deriv2, color="g")
        
    #print(result)