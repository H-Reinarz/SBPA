#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 12:22:51 2017

@author: hre070
"""

import numpy as np
from scipy.spatial.distance import euclidean
#from matplotlib import pyplot as plt




def dft_summary(freq, width=1):
        
    class_array =  np.zeros(shape=freq.shape, dtype=np.int8)
    
    middle = (class_array.shape[0]/2, class_array.shape[1]/2)
    
    for row in range(class_array.shape[0]):
        for col in range(class_array.shape[1]):
            
            distance = euclidean((row+1, col+1), middle)
            #print(distance)
            
            class_array[row, col] = distance//width
            
            
            
    #plt.imshow(class_array)
    
    class_dict = dict()
    
    for x in np.unique(class_array):
        values = freq[class_array == x]
        mean = np.mean(values)
        class_dict[x] = mean
    
    return class_dict





#in_shape = (100, 200)
#width = 1
#
#dft = np.random.randint(60, 9999, size=in_shape)/10000
#
#
#plot_values = list(dft_summary(dft).values())
#
#plt.plot(plot_values)
#    
#print(plot_values)