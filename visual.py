#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 11:46:28 2017

@author: hre070
"""
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import regionprops
from skimage.segmentation.boundaries import find_boundaries
from skimage.morphology import dilation, square


def plot_sp_labels(axes, labels, fontsize, subset=None, abs_font=False, **text_kwargs):
    '''Function that plots the label of a region to its centroid.
    Axes with appropriate background image serve as an argumunt, along with
    label image and text parameters.'''

    #increment to include 0
    labels += 1

    #convert fontsize to pt
    if not abs_font:
        wext = axes.get_window_extent()
        pt_ratio = wext.height/labels.shape[0]
        pt_size = round(fontsize*pt_ratio)
    else:
        pt_size = fontsize

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



def plot_node_attribute(axes, graph, attribute, fontsize, subset=None, **text_kwargs):
    '''Function that plots a specified attribute of a node/region to its centroid.
    Axes with appropriate background image serve as an argumunt, along with the
    graph and text parameters.'''

    #increment to include 0
    graph.seg_img += 1

    #convert fontsize to pt
    wext = axes.get_window_extent()
    pt_ratio = wext.height/graph.seg_img.shape[0]
    pt_size = round(fontsize*pt_ratio)

    if subset is not None:
        subset = set(subset)

    #iteratively plot the labels
    for reg in regionprops(graph.seg_img):
        true_label = reg.label-1

        to_plot = graph.node[true_label][attribute]

        if subset is None or to_plot in subset:
            y_pos, x_pos = reg.centroid
            axes.text(x_pos, y_pos, to_plot,
                      fontsize=pt_size, ha='center', va='center', **text_kwargs)

    #decrement to restore original data
    graph.seg_img -= 1



def show_seeds(axes, labels, seeds, marker, **plot_kwargs):
    '''Function that plots a marker to the centroid of a region (representing a seed).
    Axes with appropriate background image serve as an argumunt, along with
    label image, seeds and plot parameters.'''

    #increment to include 0
    labels += 1

    seeds = set(seeds)

    regs = [reg for reg in regionprops(labels) if (reg.label - 1) in seeds]

    y_list = [reg.centroid[0] for reg in regs]
    x_list = [reg.centroid[1] for reg in regs]

    axes.plot(x_list, y_list, marker, **plot_kwargs)

    #decrement to restore original data
    labels -= 1


def draw_boundaries(image, thickness=3):
    bounds = find_boundaries(image, mode="thick", background=0)
    bounds = bounds.astype(np.float64)
    bounds = dilation(bounds, square(thickness))
    bounds[bounds == 0] = np.nan
    return bounds

def plot_sbpa(background, segments, thickness, boundary_color, outputfile = None):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(10,10)
    ax = plt.Axes(fig, [0.,0.,1.,1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(background)
    ax.imshow(draw_boundaries(segments, thickness), cmap=boundary_color)
    ax.axis('off')
    if outputfile is not None:
        fig.savefig(outputfile)
        
def plot_sbpa_categories(background, segments, thickness, boundary_color, 
                         category_segments, category_colormap, alpha , outputfile = None):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(10,10)
    ax = plt.Axes(fig, [0.,0.,1.,1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(background)
    ax.imshow(category_segments, cmap=category_colormap, alpha=alpha)
    ax.imshow(draw_boundaries(segments, thickness), cmap=boundary_color)
    ax.axis('off')
    if outputfile is not None:
        fig.savefig(outputfile)