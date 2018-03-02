# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 23:19:31 2017

@author: Jannik
"""
import numpy as np
import time
from scipy.spatial.distance import euclidean
from ipag import IPAG

def remove_channels(image, channels_to_remove):
    '''Takes list of channel indices to remove from an image'''

    return np.delete(image, np.array(channels_to_remove),2)



def keep_channels(image, channels_to_keep):
    '''Takes list of channel indices to keep (all others will be dropped)'''

    channels = list(range(image.shape[2]))
    channels_to_remove = [c for c in channels if c not in channels_to_keep]
    return np.delete(image, np.array(channels_to_remove),2)



def normalize_image(image, deepcopy=True):
    '''Normalize Image to 0.0 - 1.0'''
    if deepcopy:
        image = np.copy(image)
    if image.ndim == 2:
        image[:,:] += abs(np.min(image[:,:]))
        image[:,:] /= np.max(image[:,:])
    elif image.ndim > 2:
        for ix in range(image.shape[2]):
            image[:,:,ix] += abs(np.min(image[:,:,ix]))
            image[:,:,ix] /= np.max(image[:,:,ix])
    return image



def merge_channels(list_of_channels):
    '''Merge list of single channel images to multi channel image'''

    zipped = np.dstack(list_of_channels)
    return zipped



def add_value(image, value):
    '''Add a value to all pixels'''

    image += value
    return image



def value_to_value(image, value_old, value_new):
    '''Set specific pixel to specific value'''

    image[image == value_old] = value_new
    return image



def highest_value_minus_one(image, value=1):
    '''Substract 1 from 255 in an image'''

    image[image == 255] -= value
    return image



def image_from_array(array, image_shape):
    '''Brings flattened array into the shape of an image'''

    return np.reshape(array, image_shape)



def one_to_three_channels(image, deepcopy = True):
    '''Makes a single channel image to a multi channel image by duplicating
    channel'''

    if deepcopy:
        image = np.copy(image)
    return merge_channels([image,image,image])

def pixel_per_cluster(graph, attribute):
    clusterDict = {}
    for node in graph:
        if '-'.join(str(layer) for layer in graph.node[node][attribute]) in clusterDict:
            clusterDict['-'.join(str(layer) for layer in graph.node[node][attribute])] += graph.node[node]['pixel_count']
        else:
            clusterDict['-'.join(str(layer) for layer in graph.node[node][attribute])] = graph.node[node]['pixel_count']
    return clusterDict


def get_max_layer(Graph, attribute):
    attr_labels = {''.join(Graph.node[node][attribute]) for node in Graph.__iter__()}

    return max({len(label) for label in attr_labels})

def filter_cluster_image(cluster_image, reference_image, filter_value):

    cluster_value_dict = {}

    unique_clusters = np.unique(cluster_image)

    for cluster in unique_clusters:
        cluster_value_dict[cluster] = np.mean(reference_image[cluster_image == cluster])

    for key, value in cluster_value_dict.items():
        if value <= filter_value:
            cluster_image[cluster_image == key] = unique_clusters.max() + 1

    return cluster_image



def get_internal_distance(feature_space, percentile=100, func=euclidean):
    '''Return a specified percentile of the maximum distance
    computable by func within the given feature space. Returns the maximum distance as default.'''
    
    assert isinstance(feature_space, IPAG.feature_space), 'Must be IPAG.feature_space!'
    
    max_list = [np.max(feature_space.array[:,col]) for col in range(feature_space.array.shape[1])]
    
    min_list = [np.min(feature_space.array[:,col]) for col in range(feature_space.array.shape[1])]
    
    return (percentile/100)*func(np.array(min_list), np.array(max_list))
    


class Stopwatch(object):
    def __init__(self):
        self.elapsed = 0.0
        self._start = None

    def start(self):
        if self._start is not None:
            raise RuntimeError('Already started')

        self._start = time.perf_counter()

    def stop(self):
        if self._start is None:
            raise RuntimeError('Not started')

        end = time.perf_counter()
        self.elapsed += end - self._start
        self._start = None

    def reset(self):
        self.elapsed = 0.0

    @property
    def running(self):
        return self._start is not None

    def __str__(self):
        m, s = divmod(self.elapsed, 60)
        h, m = divmod(m, 60)
        return "%d:%02d:%02d" % (h, m, s)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
