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
from skimage.segmentation import slic
from skimage.segmentation import quickshift
from skimage.segmentation import mark_boundaries

from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
from matplotlib import pyplot as plt

import lbp
from skimage.transform import rotate
from skimage.color import label2rgb



image = io.imread("D:/janni/Documents/Geographie/Masterarbeit/Data/ra_neu/ra2_small.jpg")

segments_slic = slic(image, n_segments=400, compactness=15, sigma=1)
print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))
f, ax = plt.subplots(figsize=(10, 10))

ax.imshow(mark_boundaries(image, segments_slic))

im_gray = rgb2gray(image)
# settings for LBP
METHOD = 'default'
radius = 8

n_points = 4 

lbp_img = lbp.local_binary_pattern(im_gray, None, 8, radius, method='default', nilbp = True)

######################################

import networkx as nx

#from bowrag import BOW_RAG, cumu_diff
import bow_rag
import bow_diff
Graph = bow_rag.BOW_RAG(segments_slic)
print(nx.info(Graph))
Graph.add_attribute('color', image, np.mean)
Graph.normalize_attribute('color', value=255)
Graph.add_attribute('var', im_gray, np.var)

fs_attrs = {'color':1, 'var':0.8}
fs1 = Graph.get_feature_space_array(fs_attrs)

from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
import scipy as sp

connectivity = nx.adjacency_matrix(Graph, weight=None)

n_clusters = 2  # number of regions

Graph.clustering('cluster1', 'AgglomerativeClustering', fs1, n_clusters=2, linkage="ward", connectivity=connectivity)
fs2 = Graph.attribute_divided_fs_arrays(fs_attrs, 'cluster1')
Graph.clustering('cluster2', 'AgglomerativeClustering', fs2, n_clusters=2, linkage="ward", connectivity=connectivity)
#ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward',
#                               connectivity=connectivity)
cluster_img = Graph.produce_cluster_image('cluster2')

from skimage import color

out = color.label2rgb(cluster_img, image, kind='avg')
out = segmentation.mark_boundaries(out, cluster_img, (0, 0, 0))

from visual import plot_node_attribute
f, ax = plt.subplots(nrows=2, figsize=(18, 18))

#ax[0].imshow(image)
ax[0].imshow(im_gray)
ax[0].imshow(out, alpha = .4)
ax[0].imshow(cluster_img, cmap=plt.cm.spectral, alpha=.2)
plot_node_attribute(ax[0], Graph, 'cluster2', 24)


ax[1].imshow(image)
#ax[1].imshow(redu_out)
#ax[1].imshow(redu, cmap=plt.cm.spectral, alpha=.2)
#plot_node_attribute(ax[1], Graph, 'redu1', 12)
f, ax = plt.subplots(figsize=(15, 15))

ax.imshow(mark_boundaries(image, cluster_img));
