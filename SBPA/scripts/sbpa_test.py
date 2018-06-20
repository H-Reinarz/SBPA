# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 16:56:40 2018

@author: Jannik
"""

# Append Repository Path
import sys
sys.path.append("D:/janni/Documents/Geographie/Masterarbeit/src")

import numpy as np

from SBPA.rgb_indices import make_rgb_indices, make_pca
from SBPA.frequency import DFTanalyzer
from SBPA.lbp import ni_lbp, radial_lbp, angular_lbp
from SBPA.histogram import Hist
from SBPA._LBP.lbp_bins import lbp_bins
from SBPA.ipag import IPAG
from SBPA.sbpa_utils import normalize_image
from SBPA.processing import *
from SBPA.metrics import *

from skimage import io
from skimage.color import rgb2gray, rgb2lab
from skimage.util import img_as_float
from skimage.segmentation import slic

# Defining Image
image = io.imread("D:/janni/Documents/Geographie/Masterarbeit/Data/ra_neu/ra2_small.jpg")
im_gray = rgb2gray(image)
im_gray = img_as_float(im_gray)


# RGB Indices
print("Start RGBI Indices")
rgbi, rgbi_dict = make_rgb_indices(image)
# Principle Component Analysis
components, components_dict = make_pca(rgbi_dict, image)


# Superpixel
print("Start Superpixel")
image = img_as_float(image)
segments_slic = slic(image, n_segments=800, compactness=12, sigma=1)
print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))


# Local Binary Pattern
print("Start LBP")
frequency = DFTanalyzer(im_gray)
frequency.fit_model(60, 5)
frequency.apply_lowpass(1, 0.01, 2)
print("LBP Radius: ", frequency.texture_radius)

METHOD = 'default'
radius = frequency.texture_radius
n_points = 8 

nilbp = ni_lbp(im_gray, n_points, radius, method="ror")
radlbp = radial_lbp(im_gray, n_points, radius*2, radius, 'ror')
anglbp = angular_lbp(im_gray, 4, radius)

rad_lbp_BINS = lbp_bins(n_points, "ror")
ang_lbp_BINS = lbp_bins(4, "default")
ni_lbp_BINS = lbp_bins(n_points, "ror")


# IPAG
print("Build IPAG")
imageLab = rgb2lab(image)
dim1Inverted = 1-components.dim1 #Principal Component needs to be inverted

Graph = IPAG(segments_slic)
Graph.add_attribute('color', normalize_image(imageLab), np.mean)
Graph.add_attribute('var', im_gray, np.var)
Graph.add_attribute("pc1", dim1Inverted, np.mean)

Graph.add_attribute('ni_lbp', nilbp, Hist, vbins=ni_lbp_BINS)
Graph.normalize_attribute('ni_lbp')
Graph.add_attribute('rad_lbp', radlbp, Hist, vbins=rad_lbp_BINS)
Graph.normalize_attribute('rad_lbp')
Graph.add_attribute('ang_lbp', anglbp, Hist, vbins=ang_lbp_BINS)
Graph.normalize_attribute('ang_lbp')


# Texture Affinity
lbp_config = {'ni_lbp':0.01, 'rad_lbp':0.01, 'ang_lbp':0.01}
lbp_fs = Graph.hist_to_fs_array(lbp_config)
Graph.cluster_affinity_attrs("texture", "KMeans", lbp_fs, n_clusters=5)


# Feature Space and metric config
fs_attrs = {'color':1.2, 'var':.3, 'pc1': .7, "texture": .7}
fs = Graph.basic_feature_space_array(fs_attrs)

attribute = 'cluster' # Attribute name for cluster results

metric_config = {"fs_var":metric(fs_variance,{}),
                 "pixel_size":metric(count_pixel,{}),
                 "multifeatures":metric(count_multi_features,{"attribute":attribute})}
fs_metrics = Graph.apply_group_metrics(fs, metric_config)
print(fs_metrics)


# Processing
stages = LogicStageDict()

start_treshholds = {'pixel_size':threshhold(1692000//20, '>')} # If FS greater than threshold
stages['start'] = LogicStage(start_treshholds, algorithm='KMeans', n_clusters=2)

kmeans_treshholds = {'pixel_size':threshhold(1692000//4, '>')} 
stages['kmeans'] = ClusterStage(start_treshholds, algorithm='KMeans', n_clusters=2)

agglo_treshholds = {'pixel_size':threshhold(1692000//4, '<=')}
stages['agglo'] = ClusterStage(start_treshholds, algorithm='AgglomerativClustering', n_clusters=2)

stages['split'] = SplittingStage()

# Linked Stage list
# After every stage start with entry_point again
stages.link_stages('start', 'kmeans') #(stage, if true do kmeans, if false stop)
stages.link_stages('kmeans', 'split', 'agglo') #(stage, if true do split, if false do agglo)
stages.link_stages('agglo', 'split') #(stage, if true do split, if false do nothing)

stages.initiate_stages()

print('Start dynamic clustering:')
dynamic_clustering(Graph, fs_attrs, attribute, metric_config, stages['start'], stages['split'])