# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 22:30:53 2017

@author: Jannik
"""

import utils as u
import numpy as np
from skimage.util import img_as_float
import copy
import pandas as pd
from sklearn.decomposition import PCA
R = 0
G = 1
B = 2

# Green Leaf Index
def GLI(image):
    return (2*image[:,:,G] - image[:,:,R] - image[:,:,B]) / (2*image[:,:,G] + image[:,:,R] + image[:,:,B])

# Visible Atmospherically Resistant Index
def VARI(image):
    return (image[:,:,G] - image[:,:,R]) / (image[:,:,G] + image[:,:,R] - image[:,:,B])

# Visible Vegetation Index
def VVI(image):
    return (1 - abs((image[:,:,R] - 30) / (image[:,:,R] + 30))) * (1 - abs((image[:,:,G] - 50) / (image[:,:,G] + 50))) * (1 - abs((image[:,:,B] - 1) / (image[:,:,B] + 1)))

# Normalized difference turbidity index
def NDTI(image):
    return (image[:,:,R] - image[:,:,G]) / (image[:,:,R] + image[:,:,G])

# Redness index
def RI(image):
    return (image[:,:,R]**2) / (image[:,:,B] * image[:,:,G]**3)

# Soil Color Index
def CI(image):
    return (image[:,:,R] - image[:,:,G]) / (image[:,:,R] + image[:,:,G])

# Brightness Index
def BI(image):
    return (np.sqrt((image[:,:,R]**2 + image[:,:,G]**2 + image[:,:,B]*2) / 3))

# Spectral Slope Saturation Index
def SI(image):
    return (image[:,:,R] - image[:,:,B]) / (image[:,:,R] + image[:,:,B])

# Primary colours Hue Index
def HI(image):
    return ((2*image[:,:,R] - image[:,:,G] - image[:,:,B]) / (image[:,:,G] - image[:,:,B]))
    
# Triangular greeness index
def TGI(image):
    return -0.5*(190*(image[:,:,R] - image[:,:,G]) - 120 * (image[:,:,R] - image[:,:,B]))

# Normalized green red difference index
def NGRDI(image):
    return ((image[:,:,G] - image[:,:,R]) / (image[:,:,G] + image[:,:,R]))

def MakeRgbIndices(img, normalize = True, zeros = 1):
    if img.dtype == "float64":
        raise TypeError("Image has to be unsigned integer for creating RGB Indices!")
    
    image = np.copy(img)
    image = u.ZerosToOne(image, zeros)
    image = img_as_float(image)    
    
    rgbiDict = {"gli":GLI(image),
                "vvi" : VVI(image),
                "ntdi" : NDTI(image),
                "ci" : CI(image),
                "bi" : BI(image),
                "si" : SI(image),
                "tgi" : TGI(image),
                "ngrdi" : NGRDI(image)}
    
    if normalize:
        for key, value in rgbiDict.items():
            rgbiDict[key] = u.NormalizeImage(rgbiDict[key])
    
    rgbi = type("rgbi", (), rgbiDict)
    p = rgbi()
    return p, rgbiDict

def MakePCA(di, image, components = 3):
    d = copy.deepcopy(di)
    
    for key, value in d.items():
        d[key] = d[key].flatten()
    df = pd.DataFrame(d)
    
    pca = PCA(n_components=components).fit(df)
    pca_reduced = pca.transform(df)
    
    dimDict = {}
    
    for i in range(0,components):
        dimDict["dim"+str(i+1)] = u.NormalizeImage(u.ImageFromArray(pca_reduced[:,i], image))
    
    comp = type("components", (), dimDict)
    p = comp()
    return p, dimDict