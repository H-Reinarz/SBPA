# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 12:42:57 2017

@author: Jannik
"""


# "H:/Geography/MASTERARBEIT/Data/ResearchArea/RA1/dsmRA1.tif"
# "H:/Geography/MASTERARBEIT/Data/ResearchArea/RA1/orthoClipRA1.tif"



from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import importlib
import skimage.segmentation.slic6Dimensions

from skimage.data import astronaut
from skimage.color import label2rgb
from skimage.filters import sobel
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from skimage.segmentation import quickshift

importlib.reload(skimage.segmentation.slic6Dimensions)

img = io.imread("H:/Geography/MASTERARBEIT/Data/ResearchArea/RA1/orthoClipRA1_badRes.jpg")
#img = io.imread("D:/Bilder/solidBlack.jpg")


img2 = io.imread("H:/Geography/MASTERARBEIT/Data/ResearchArea/RA1/dsmRA1_test16_resampled.png")
#img2 = io.imread("D:/Bilder/solidBlack_Square.jpg")
"""
for row in range(0, img2.shape[0]):
    for col in range(0, img2.shape[1]):
        if img2[row, col] <= 30000:
            img2[row, col] = 0
            img[row, col, 0] = 0
            img[row, col, 1] = 0
            img[row, col, 2] = 0
"""
img = img_as_float(img[::2, ::2])
img2 = img_as_float(img2[::2, ::2])



r = img[:,:,0]
g = img[:,:,1]
b = img[:,:,2]
bw = img2[:,:]

zipped = np.dstack((r, g, b, bw))

#segments_slic = skimage.segmentation.slic6Dimensions.slic6(zipped, n_segments=150, compactness=.2, sigma=1)
segments_slic = quickshift(zipped, kernel_size=8, max_dist=16, ratio=0.5)
#out1 = label2rgb(segments_slic, zipped, kind='avg')

f, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))

axes.imshow(mark_boundaries(img, segments_slic))
#axes.imshow(out1)
print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))
"""
img = img[:,:,:]

#zipped = np.dstack((, g, b))



segments_slic = slic6(img, n_segments=150, compactness=10, sigma=1)

f, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))

axes.imshow(mark_boundaries(img, segments_slic))
print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))
"""