# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 17:18:49 2017

@author: Jannik
"""
import rasterio
from rasterio.transform import from_origin
import numpy as np

class Output:
    def __init__(self, crsTif, noCrsImg):
        self.crsTif = crsTif
        self.noCrsImg = noCrsImg
    
    def WriteCrs(self, filename, intern = False):
        dataset = rasterio.open(self.crsTif)
        x1 = dataset.bounds[0] # left
        x2 = dataset.bounds[2] # right
        y1 = dataset.bounds[1] # bottom
        y2 = dataset.bounds[3] # top
        crs = dataset.crs.to_string()
        dataset.close()
        
        if intern: # does not work
            array = self.noCrsImg.astype('uint8')
            maskData = array
            maskRaster = maskData
            x = np.linspace(x1, x2, maskData.shape[2])
            y = np.linspace(y1, y2, maskData.shape[1])
            res = (x[-1] - x[0]) / maskData.shape[2]
            transform = from_origin(x[0] - res / 2, y[-1] + res / 2, res, res)
        
            new_dataset = rasterio.open(filename, "w", driver="GTiff",
                                height = maskData.shape[1], width = maskData.shape[2],
                                count=1, dtype = maskRaster.dtype,
                                crs=crs, transform = transform)
            new_dataset.write(maskRaster[0], 1) #stores band one. To Do: FInd out how to store every band
            new_dataset.close()
        else:
            maskData = rasterio.open(self.noCrsImg)
            maskRaster = maskData.read()
            x = np.linspace(x1, x2, maskData.width)
            y = np.linspace(y1, y2, maskData.height)
            res = (x[-1] - x[0]) / maskData.width
            transform = from_origin(x[0] - res / 2, y[-1] + res / 2, res, res)
        
            new_dataset = rasterio.open(filename, "w", driver="GTiff",
                                height = maskData.height, width = maskData.width,
                                count=1, dtype = maskRaster.dtype,
                                crs=crs, transform = transform)
            new_dataset.write(maskRaster[0], 1) #stores band one. To Do: FInd out how to store every band
            new_dataset.close()



