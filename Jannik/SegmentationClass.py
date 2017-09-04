# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 12:47:11 2017

@author: Jannik
"""
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb
import skimage.segmentation.slic6Dimensions
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from skimage.segmentation import quickshift

class Segmentation:
    def __init__(self, rgb, height, asFloat = True):
        self.rgb = io.imread(rgb)
        self.height = io.imread(height)
        self.rgbOriginal = self.rgb
        self.heightOriginal = self.height
        self.isFloat = asFloat
        self.image4D = ""
        if(self.isFloat):
            self.ToFloat()
            self.Make4D()
        self.segments = ""
        self.labelAvg = ""
        self.boundaries = ""
        
    def Print(self):
        plt.imshow(self.rgb)
        
    def ToFloat(self):
        self.rgb = img_as_float(self.rgb[::2, ::2])
        self.height = img_as_float(self.height[::2, ::2])
        self.isFloat = True
        
    def Make4D(self):
            r = self.rgb[:,:,0]
            g = self.rgb[:,:,1]
            b = self.rgb[:,:,2]
            bw = self.height[:,:]
            self.image4D = np.dstack((r, g, b, bw))
            
    def Default(self):
        self.rgb = self.rgbOriginal
        self.height = self.heightOriginal
        
    def Slic6(self, _Use4D = False, _n_segments = 150, _compactness = 0.2, _sigma = 1, imgAlt = ""):
        if _Use4D:
            img = self.image4D
        else:
            img = self.rgb
        
        if len(imgAlt) > 0:
            img = imgAlt
        
        self.segments = skimage.segmentation.slic6Dimensions.slic6(img, n_segments = _n_segments, compactness = _compactness, sigma = _sigma)
        
    def Quickshift(self, _Use4D = False, _kernelSize = 8, _maxDist = 16, _ratio = 0.5, imgAlt = ""):
        if _Use4D:
            img = self.image4D
        else:
            img = self.rgb
        
        if len(imgAlt) > 0:
            img = imgAlt
            
        self.segments = quickshift(img, kernel_size = _kernelSize, max_dist = _maxDist, ratio = _ratio)
        
    def Label2Rgb(self, _Use4D = False):
        if _Use4D:
            img = self.image4D
        else:
            img = self.rgb
        self.labelAvg = label2rgb(self.segments, img, kind='avg')
    
    def MarkBoundaries(self, mask = True):
        maskImageBg = np.zeros((self.rgb.shape[0], self.rgb.shape[1], 3), dtype=np.double)
        if mask:
            self.boundaries = mark_boundaries(maskImageBg, self.segments, color=(1, 1, 1))
        else:
            self.boundaries = mark_boundaries(self.rgb, self.segments, color=(1, 1, 1))
    
    def ImgSave(self, fname, folder):
        io.imsave(folder+"/"+fname+"_segments"+".png", self.boundaries, plugin=None)
        io.imsave(folder+"/"+fname+"_avgColor"+".png", self.labelAvg, plugin=None)
        io.imsave(folder+"/"+fname+"_labels"+".png", self.segments, plugin=None)

        

    
    
    
    
    