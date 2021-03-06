# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 23:19:31 2017

@author: Jannik
"""
import numpy as np
import time
from sklearn.cluster import KMeans


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


def categorize(segments, image, nc):
    
    ld_clusters = {}
    
    for label in np.unique(segments):
        mask = segments == label
        
        ld_clusters[label] = [np.mean(image[:,:,z][mask]) for z in range(3)]
        
            
    ld_feature_space = np.array(list(ld_clusters.values()))
    
    cluster_obj = KMeans(n_clusters=nc)
    
    cluster_obj.fit(ld_feature_space)
    
    ld_results = dict(zip(ld_clusters, cluster_obj.labels_))
    
    result = np.zeros_like(segments)
    
    for label in np.unique(segments):
        mask = segments == label 
        
        result[mask] = ld_results[label]
        
    return result


##Initialize log file and mock print function
class DoubleLogStream(object):
    def __init__(self, file, console):
        self.file = file
        self.console = console

    def write(self, obj):
        self.file.write(obj)
        self.console.write(obj)

    def flush(self):
        self.file.flush()
        self.console.flush()

        
    
    

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
