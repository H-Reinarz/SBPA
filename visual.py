#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 11:46:28 2017

@author: hre070
"""
from skimage.measure import regionprops



def plot_sp_labels(axes, labels, fontsize, **text_kwargs):
    
    wext = axes.get_window_extent()
    
    pt_ratio = wext.height/labels.shape[0]
    print(pt_ratio)
    
    pt_size = round(fontsize*pt_ratio)
    
    for reg in regionprops(labels):
        y, x = reg.centroid
        axes.text(x, y, reg.label, fontsize=pt_size, ha='center', va='center', **text_kwargs)