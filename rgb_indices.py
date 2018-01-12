# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 22:30:53 2017

@author: Jannik
"""

import sbpa_utils as u
import numpy as np
from skimage.util import img_as_float
import copy
import pandas as pd
from sklearn.decomposition import PCA
R = 0
G = 1
B = 2


def GLI(image):
    '''Returns Green Leaf Index'''
    return (2*image[:,:,G] - image[:,:,R] - image[:,:,B]) / (2*image[:,:,G] + image[:,:,R] + image[:,:,B])
 
def VARI(image):
    '''Returns Visible Atmospherically Resistant Index'''
    return (image[:,:,G] - image[:,:,R]) / (image[:,:,G] + image[:,:,R] - image[:,:,B])
 
def VVI(image):
    '''Returns Visible Vegetation Index'''
    return (1 - abs((image[:,:,R] - 30) / (image[:,:,R] + 30))) * (1 - abs((image[:,:,G] - 50) / (image[:,:,G] + 50))) * (1 - abs((image[:,:,B] - 1) / (image[:,:,B] + 1)))
 
def NDTI(image):
    '''Returns Normalized difference turbidity index'''
    return (image[:,:,R] - image[:,:,G]) / (image[:,:,R] + image[:,:,G])

def RI(image):
    '''Returns Redness index'''
    return (image[:,:,R]**2) / (image[:,:,B] * image[:,:,G]**3)

def CI(image):
    '''Returns Soil Color Index'''
    return (image[:,:,R] - image[:,:,G]) / (image[:,:,R] + image[:,:,G])
 
def BI(image):
    '''Returns Brightness Index'''
    return (np.sqrt((image[:,:,R]**2 + image[:,:,G]**2 + image[:,:,B]*2) / 3))

def SI(image):
    '''Returns Spectral Slope Saturation Index'''
    return (image[:,:,R] - image[:,:,B]) / (image[:,:,R] + image[:,:,B])

def HI(image):
    '''Returns Primary colours Hue Index'''
    return ((2*image[:,:,R] - image[:,:,G] - image[:,:,B]) / (image[:,:,G] - image[:,:,B]))
    
def TGI(image):
    '''Returns Triangular greeness index'''
    return -0.5*(190*(image[:,:,R] - image[:,:,G]) - 120 * (image[:,:,R] - image[:,:,B]))

def NGRDI(image):
    '''Returns Normalized green red difference index'''
    return ((image[:,:,G] - image[:,:,R]) / (image[:,:,G] + image[:,:,R]))



def make_rgb_indices(img, normalize = True, zeros = 1):
    '''Returns an indices object and a dictionary of gli, vvi, ntdi, ci,
    bi, si, tgi, ngrdi'''
    
    if img.dtype == "float64":
        raise TypeError("Image has to be unsigned integer for creating RGB Indices!")
    
    image = np.copy(img)
    image = u.value_to_value(image,0, zeros)
    image = img_as_float(image)    
    
    rgbi_dict = {"gli":GLI(image),
                "vvi" : VVI(image),
                "ntdi" : NDTI(image),
                "ci" : CI(image),
                "bi" : BI(image),
                "si" : SI(image),
                "tgi" : TGI(image),
                "ngrdi" : NGRDI(image)}
    
    if normalize:
        for key, value in rgbi_dict.items():
            rgbi_dict[key] = u.normalize_image(rgbi_dict[key])
    
    rgbi = type("rgbi", (), rgbi_dict)
    p = rgbi()
    return p, rgbi_dict



def make_pca(di, image, components = 3):
    '''Takes a list (Currently Dictionary with "Name": Array)of single-channel 
    images and performs a principal component analysis. TODO: Get rid of 
    pandas)'''
    
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