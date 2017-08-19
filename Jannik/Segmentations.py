# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 16:42:47 2017

@author: Jannik
"""

import SegmentationClass as s
import OutputClass as o
import numpy as np
import os


originalTif = "H:/Geography/MASTERARBEIT/Data/ResearchArea/RA1/orthoClipRA1_badRes.tif"
rgb = "H:/Geography/MASTERARBEIT/Data/ResearchArea/RA1/orthoClipRA1_badRes.jpg"
height = "H:/Geography/MASTERARBEIT/Data/ResearchArea/RA1/dsmRA1_test16_resampled.png"
wd = "H:/Geography/MASTERARBEIT/Data/ResearchArea/RA1/Segmentations"
tiffname = os.path.join(wd, 'rasterio.tif')

slic1 = s.Segmentation(rgb, height)
slic1.Slic6(False, 150, 10, 1)
slic1.Label2Rgb()
slic1.MarkBoundaries(False)
slic1.ImgSave("slic-150-10-1", wd)


slic1.Slic6(True, 150, .2, 1)
slic1.Label2Rgb(True)
slic1.MarkBoundaries(False)
slic1.ImgSave("slic-150-02-1", wd)

quick = s.Segmentation(rgb, height)
quick.Quickshift(False, 8, 16, 0.5)
quick.Label2Rgb()
quick.MarkBoundaries(False)
quick.ImgSave("quick-8-16-05", wd)

quick = s.Segmentation(rgb, height)
quick.Quickshift(False, 16, 32, 0.5)
quick.Label2Rgb()
quick.MarkBoundaries(False)
quick.ImgSave("quick-16-32-05", wd)

quick = s.Segmentation(rgb, height)
quick.Quickshift(False, 32, 32, 0.5)
quick.Label2Rgb()
quick.MarkBoundaries(False)
quick.ImgSave("quick-16-32-05", wd)











"""
crs1 = o.Output(originalTif, "H:/Geography/MASTERARBEIT/Data/ResearchArea/RA1/Segmentations/slic-150-10-1_labels.png")
crs1.WriteCrs("H:/Geography/MASTERARBEIT/Data/ResearchArea/RA1/Segmentations/slic-150-10-1_labelsCrs.tif", False)
"""
