# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 22:30:53 2017

@author: Jannik
"""

import numpy as np
from enum import Enum

class C(Enum):
    Red = 0
    Green = 1
    Blue = 2
    
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