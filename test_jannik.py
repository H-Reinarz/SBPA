# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 10:38:45 2017

@author: Jannik
"""

import sys
sys.path.append("D:/janni/Documents/Geographie/Masterarbeit/src/github")
sys.path.append("D:/janni/Documents/Geographie/Masterarbeit/src/github/_LBP")

from skimage import io
from skimage import data, segmentation
import rgb_indices as rgb
import utils as u
import numpy as np
from skimage.util import img_as_float


from skimage.color import rgb2gray
from matplotlib import pyplot as plt


image = io.imread("D:/janni/Documents/Geographie/Masterarbeit/Data/ra_neu/ra2_small.jpg")

image = image.astype('int32')
#image = img_as_float(image)

#image = u.AddValue(image, 0.1)
image = u.ZerosToOne(image, 2)
#image = img_as_float(image)

gli = rgb.GLI(image)
gli = u.NormalizeImage(gli)
f, ax = plt.subplots(figsize=(7, 7))
ax.imshow(gli)

gli = rgb.VARI(image)
#gli = u.ZerosToOne(gli, 0.001)
gli = u.NormalizeImage(gli)
f, ax = plt.subplots(figsize=(7, 7))
ax.imshow(gli)

bla = np.zeros_like(image[::,::,0], dtype="float64")
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        bla[i,j] = (image[i,j,1] - image[i,j,0]) / (image[i,j,1] + image[i,j,0] - image[i,j,2])
        if np.isinf(bla[i,j]):
            print((image[i,j,1] - image[i,j,0]), " / ",(image[i,j,1] + image[i,j,0] - image[i,j,2]) , " = ", bla[i,j])


gli = rgb.VVI(image)
gli = u.NormalizeImage(gli)
f, ax = plt.subplots(figsize=(7, 7))
ax.imshow(gli)

gli = rgb.NDTI(image)
gli = u.NormalizeImage(gli)
f, ax = plt.subplots(figsize=(7, 7))
ax.imshow(gli)

gli = rgb.RI(image)
gli = u.NormalizeImage(gli)
f, ax = plt.subplots(figsize=(7, 7))
ax.imshow(gli)

gli = rgb.CI(image)
gli = u.NormalizeImage(gli)
f, ax = plt.subplots(figsize=(7, 7))
ax.imshow(gli)

gli = rgb.BI(image)
gli = u.NormalizeImage(gli)
f, ax = plt.subplots(figsize=(7, 7))
ax.imshow(gli)

gli = rgb.SI(image)
gli = u.NormalizeImage(gli)
f, ax = plt.subplots(figsize=(7, 7))
ax.imshow(gli)

gli = rgb.HI(image)
gli = u.NormalizeImage(gli)
f, ax = plt.subplots(figsize=(7, 7))
ax.imshow(gli)

gli = rgb.TGI(image)
gli = u.NormalizeImage(gli)
f, ax = plt.subplots(figsize=(7, 7))
ax.imshow(gli)

gli = rgb.NGRDI(image)
gli = u.NormalizeImage(gli)
f, ax = plt.subplots(figsize=(7, 7))
ax.imshow(gli)        
     