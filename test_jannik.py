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


from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
from matplotlib import pyplot as plt

import pandas as pd

image = io.imread("D:/janni/Documents/Geographie/Masterarbeit/Data/ra_neu/ra2_small.jpg")
#image = io.imread("D:/janni/Documents/Geographie/Masterarbeit/Data/ResearchArea/RA1/orthoRA.jpg")

#image = image.astype('int32')
#image = img_as_float(image)

#image = u.AddValue(image, 1)
image = u.ZerosToOne(image, 1)
image = img_as_float(image)

gli = rgb.GLI(image)
vvi = rgb.VVI(image)
ntdi = rgb.NDTI(image)
ci = rgb.CI(image)
bi = rgb.BI(image)
si = rgb.SI(image)
tgi = rgb.TGI(image)
ngrdi = rgb.NGRDI(image)

gli = u.NormalizeImage(gli)
vvi = u.NormalizeImage(vvi)
ntdi = u.NormalizeImage(ntdi)
ci = u.NormalizeImage(ci)
bi = u.NormalizeImage(bi)
si = u.NormalizeImage(si)
tgi = u.NormalizeImage(tgi)
ngrdi = u.NormalizeImage(ngrdi)

df = pd.DataFrame({'gli':gli.flatten(), 'vvi':vvi.flatten(), 'ntdi':ntdi.flatten(), 
                   'ci':ci.flatten(), 'bi':bi.flatten(), 'si':si.flatten(), 
                   'tgi':tgi.flatten(), 'ngrdi':ngrdi.flatten(),
                   'r':image[:,:,0].flatten(), 'g':image[:,:,1].flatten(), 'b':image[:,:,2].flatten()})



from sklearn.decomposition import PCA
pca = PCA(n_components=3).fit(df)
pca_reduced = pca.transform(df)

dim1 = u.ImageFromArray(pca_reduced[:,0], image)
dim2 = u.ImageFromArray(pca_reduced[:,1], image)
dim3 = u.ImageFromArray(pca_reduced[:,2], image)

#stack = u.MergeChannels([dim1,dim2,dim3])

#stack = u.NormalizeImage(stack)
dim1 = u.NormalizeImage(dim1)
#dim2 = u.NormalizeImage(dim2)
#dim3 = u.NormalizeImage(dim3)
#
#f, ax = plt.subplots(nrows= 4, figsize=(25, 50))
#ax[0].imshow(stack, cmap='gray')
#ax[1].imshow(dim1, cmap='gray')
#ax[2].imshow(dim2, cmap='gray')
#ax[3].imshow(dim3, cmap='gray')




imageHsv = rgb2hsv(image)
dim1Inverted = 1-dim1
imageHsv[:,:,2] = dim1Inverted
imageHsv = hsv2rgb(imageHsv)

f, ax = plt.subplots(figsize=(17, 17))
ax.imshow(imageHsv)

#reduced_data = pd.DataFrame(pca_reduced, columns = ['Dimension 1', 'Dimension 2'])

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
segments_slic = slic(imageHsv, n_segments=700, compactness=20, sigma=1)
print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))

f, ax = plt.subplots(figsize=(17, 17))
ax.imshow(mark_boundaries(imageHsv, segments_slic), cmap='gray');

#stack2 = u.MergeChannels([vvi,bi,tgi])


"""



segments_slic = slic(stack2, n_segments=100, compactness=15, sigma=1)
print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))

f, ax = plt.subplots(figsize=(17, 17))
ax.imshow(mark_boundaries(image, segments_slic));
"""