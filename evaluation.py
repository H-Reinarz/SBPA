#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 11:26:03 2018

@author: hre070
"""

import numpy as np
from skimage.segmentation import find_boundaries
from statistics import mean

def single_label_jaccard(reference, label, segmentation, only_max=True):
    
    assert reference.shape == segmentation.shape
    
    mask = reference == label
    
    values = np.unique(segmentation[mask])
    
    jaccard_dict = {}
    
    for value in values:
        value_mask = segmentation == value
        
        intersection = np.logical_and(mask, value_mask)
    
        union = np.logical_or(mask, value_mask)

        jaccard_dict[value] = intersection.sum() / float(union.sum())
    
    if only_max:
        return max(jaccard_dict.values())
    else:
        return jaccard_dict



def get_boundary_length(reference, label, mode='inner'):
    mask = reference == label
    
    boundaries = find_boundaries(mask, mode=mode, background=False)

    return(boundaries.sum())



def get_accuracy(reference, segmentation, mean_func=mean):
    
    assert reference.shape == segmentation.shape
    
    labels = np.unique(reference)
    
    circumference = 2*sum(reference.shape)
    
    print('Circumf.:', circumference)
    
    boundary_dict = {label: get_boundary_length(reference, label)/circumference for label in labels}
    
    print('Boundaries:',boundary_dict)
    
    central_bl = mean_func(list(boundary_dict.values()))
    
    print('mean length:', central_bl)
    
    diff_dict = {label: central_bl + abs(bl - central_bl) for label, bl in boundary_dict.items()}
    
    print('diff dict: ',diff_dict)
    
    results = {label: diff * single_label_jaccard(reference, label, segmentation) for label, diff in diff_dict.items()}
    
    return results
    


ref = np.array([[0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0],
                [0,0,1,1,1,1,1,0,0,0],
                [0,0,1,1,1,1,1,0,0,0],
                [0,0,1,1,1,1,1,0,0,0],
                [0,0,1,1,1,1,1,0,0,0],
                [0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0]])

#seg = np.array([[0,0,0,0,0,0,0,0,0,0],
#                [0,0,0,0,0,0,0,0,0,0],
#                [0,0,1,1,1,1,1,0,0,0],
#                [0,0,1,1,1,1,1,0,0,0],
#                [0,0,1,1,1,1,1,0,0,0],
#                [0,0,1,1,1,1,1,0,0,0],
#                [0,0,0,0,0,0,0,0,0,0],
#                [0,0,0,0,0,0,0,0,0,0],
#                [0,0,0,0,0,0,0,0,0,0],
#                [0,0,0,0,0,0,0,0,0,0]])



    
seg = np.array([[0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0],
                [0,0,1,1,0,0,0,0,0,0],
                [0,0,1,1,1,1,1,0,0,0],
                [0,0,1,1,2,2,2,0,0,0],
                [0,0,1,1,2,2,2,0,0,0],
                [0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0]])

    
#seg = np.array([[0,0,0,0,0,0,0,0,0,0],
#                [0,0,0,0,0,0,0,0,0,0],
#                [0,0,0,0,0,0,0,0,0,0],
#                [0,0,0,0,0,0,0,0,0,0],
#                [0,0,0,0,0,0,0,0,0,0],
#                [0,0,0,0,0,0,0,0,0,0],
#                [0,0,0,0,0,0,0,0,0,0],
#                [0,0,0,0,0,0,0,0,0,0],
#                [0,0,0,0,0,0,0,0,0,0],
#                [0,0,0,0,0,0,0,0,0,0]])
#    
    
print(single_label_jaccard(ref, 1, seg, only_max=False))

print(get_accuracy(ref, seg))