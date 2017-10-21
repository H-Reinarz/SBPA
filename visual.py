#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 11:46:28 2017

@author: hre070
"""
from skimage.measure import regionprops



def plot_sp_labels(axes, labels, fontsize, subset=None, **text_kwargs):
    '''Function that plots the label of a region to its centroid.
    Axes with appropriate background image serve as an argumunt, along with
    label image and text parameters.'''

    #increment to include 0
    labels += 1

    #convert fontsize to pt
    wext = axes.get_window_extent()
    pt_ratio = wext.height/labels.shape[0]
    pt_size = round(fontsize*pt_ratio)


    if subset is not None:
        subset = set(subset)

    #iteratively plot the labels
    for reg in regionprops(labels):
        true_label = reg.label-1

        if subset is None or true_label in subset:
            y_pos, x_pos = reg.centroid
            axes.text(x_pos, y_pos, true_label,
                      fontsize=pt_size, ha='center', va='center', **text_kwargs)

    #decrement to restore original data
    labels -= 1



def show_seeds(axes, labels, seeds, marker, **plot_kwargs):
    '''Function that plots a marker to the centroid of a region (representing a seed).
    Axes with appropriate background image serve as an argumunt, along with
    label image, seeds and plot parameters.'''

    #increment to include 0
    labels += 1

    seeds = set(seeds)

    regs = [reg for reg in regionprops(labels) if reg.label in seeds]

    y_list = [reg.centroid[0] for reg in regs]
    x_list = [reg.centroid[1] for reg in regs]

    axes.plot(x_list, y_list, marker, **plot_kwargs)

    #decrement to restore original data
    labels -= 1
