#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Aug  7 14:25:44 2017

@author: hre070
'''

#Imports
import statistics as stats
from collections import namedtuple, Counter
import copy
#import networkx as nx
#from itertools import repeat
from skimage.future import graph
import networkx as nx
from numpy.lib.stride_tricks import as_strided
from scipy import ndimage as ndi
from graph import _add_edge_filter
import numpy as  np
from bow_container import hist
from skimage.future.graph import RAG
from skimage.measure import regionprops
import sklearn.cluster
import sys
sys.path.append("H:/Geography/MASTERARBEIT/src/github/_LBP")
import _graph


def calc_attr_value(*, array, func, **kwargs):
    '''Helper function to apply a given function to
    a numpy array (i.e. an image) and return the result.
    If the array has multiple dimensions, a list of values is returned.'''
    #account for multi channels: one hist per channel
    if len(array.shape) == 2:
        result = [func(array[:, dim], **kwargs) for dim in range(array.shape[1])]
    else:
        result = func(array, **kwargs)

    return result




#Subclass of RAG specified for BOW classification
class BOW_RAG(nx.Graph):
    '''Subclass of the 'region adjacency graph' (RAG) in skimage to accomodate for
    dynamic attribute assignment, neighbourhood weighting and node clustering.'''

    config = namedtuple('AttributeConfig', ['img', 'func', 'kwargs'])

    def __init__(self, seg_img, **attr):
        
        #Call the RAG constructor
        #super().__init__(label_image=seg_img, connectivity=1, data=None, **attr)
        
        super(BOW_RAG, self).__init__(None, **attr)
        if self.number_of_nodes() == 0:
            self.max_id = 0
        else:
            self.max_id = max(self.nodes())

        if seg_img is not None:
            fp = ndi.generate_binary_structure(seg_img.ndim, 1)
            # In the next ``ndi.generic_filter`` function, the kwarg
            # ``output`` is used to provide a strided array with a single
            # 64-bit floating point number, to which the function repeatedly
            # writes. This is done because even if we don't care about the
            # output, without this, a float array of the same shape as the
            # input image will be created and that could be expensive in
            # memory consumption.
            
            edgeSet = set()
            
            #seg_img = seg_img.astype(int, copy = False)
            
            ndi.generic_filter(
                seg_img,
                function=_graph._graphLabels,
                footprint=fp,
                mode='nearest',
                output=as_strided(np.empty((1,), dtype=np.float_),
                                  shape=seg_img.shape,
                                  strides=((0,) * seg_img.ndim)), extra_arguments=(edgeSet,))
        print(len(edgeSet))
        for edge in edgeSet:
            if not self.has_edge(edge[0], edge[1]):
                self.add_edge(edge[0], edge[1])

        #Store seg_img as attribute
        self.seg_img = seg_img


        #Node attribute reference information
        self.attr_func_configs = {}
        self.attr_norm_val = {}

        #Init edge weight statistics
        self.edge_weight_stats = {}


        #Set indipendent node attributes
        for node in self.__iter__():
            #get color values for super pixel
            label_mask = self.seg_img == node

            #Assign attributes to node
            self.node[node].update({'labels': [node],
                                    'pixel_count': seg_img[label_mask].size})


    def add_attribute(self, name, image, function, **func_kwargs):
        '''Adds an attribute ('name') to each node by calling 'calc_attr_value()'
        on the subset of the image that is represented by the node.'''

        self.attr_func_configs[name] = BOW_RAG.config(image, function, func_kwargs)

        #Set node attributes
        for node in self.__iter__():
            #get color values for super pixel
            label_mask = self.seg_img == node
            masked_image = image[label_mask]

            attr_value = calc_attr_value(array=masked_image, func=function, **func_kwargs)

            #Assign attributes to node
            self.node[node].update({name:attr_value})



    def add_attribute_from_lookup(self, new_attribute, attribute, lookup_dict):
        '''Assign a new node attribute with values from the provided
        look up dictionary corresponding to an existing attribute.'''

        for node in self.__iter__():
            key = self.node[node][attribute]
            self.node[node].update({new_attribute: lookup_dict[key]})




    def add_regionprops(self):
        '''Function to assign geometric properties of the represented region
        as node attributes. IN DEVELOPMENT!'''

        self.seg_img += 1

        for reg in regionprops(self.seg_img):
            self.node[reg.label-1]["Y"] = round(reg.centroid[0]/self.seg_img.shape[0], 3)
            self.node[reg.label-1]["X"] = round(reg.centroid[1]/self.seg_img.shape[1], 3)

        self.seg_img -= 1


    def normalize_attribute(self, attribute, value=None):
        '''Normalize a node attribute with a given denominator.'''

        self.attr_norm_val[attribute] = value

        for node in self.__iter__():
            if isinstance(self.node[node][attribute], list):
                for index, element in enumerate(self.node[node][attribute]):
                    if isinstance(element, hist):
                        element.normalize(self.node[node]['pixel_count'])
                    else:
                        self.node[node][attribute][index] = element/value

            else:
                if isinstance(self.node[node][attribute], hist):
                    self.node[node][attribute].normalize(self.node[node]['pixel_count'])
                else:
                    self.node[node][attribute] /= value




    def delete_attributes(self, attribute):
        '''Delete a given attribute.'''
        for node in self.__iter__():
            del self.node[node][attribute]


    def deepcopy_node(self, node):
        '''Return a deep copy of a node dictionary.'''
        #create mutable copy of the node for calculation
        return copy.deepcopy(self.node[node])



    def calc_edge_weights(self, weight_func):
        '''Apply a given weighting function to all edges.'''

        #Iterate over edges and calling weight_func on the nodes
        for node1, node2, data in self.edges_iter(data=True):
            data.update(weight_func(self, node1, node2))


    def get_edge_weight_list(self, attr_label='weight'):
        '''Return a sorted value list of a given edge attribute.'''
        return sorted(list(data[attr_label] for node1, node2, data in self.edges(data=True)))


    def calc_edge_weight_stats(self, attr_label='weight'):
        '''Perform descriptive stats on a given edge attribute.
        Result is stored as a graph attribute.'''
        weight_list = self.get_edge_weight_list(attr_label)

        self.edge_weight_stats['min'] = min(weight_list)
        self.edge_weight_stats['max'] = max(weight_list)
        self.edge_weight_stats['mean'] = stats.mean(weight_list)
        self.edge_weight_stats['median'] = stats.median(weight_list)
        self.edge_weight_stats['stdev'] = stats.stdev(weight_list)


    def get_edge_weight_percentile(self, perc, attr_label='weight', as_threshhold=False):
        '''Return the given percentile value for the value list af a specified attribute.
        When 'as_threshhold' is true, the mean of the percentile value
        and the next value is returned.'''
        weight_list = self.get_edge_weight_list(attr_label)

        index = round(len(weight_list)*(perc/100))

        if as_threshhold:
            result = (weight_list[index] +  weight_list[index+1]) /2
            return result
        else:
            return weight_list[index]


    def get_feature_space_array(self, attr_config, hist_func=lambda x: x):
        '''Arange a specification of attributes into an array that contains
        one row per node. It serves as data points in feature space for clustering operations.'''

        weight_list = list()
        array_list = list()

        for attr, weight in attr_config.items():
            if isinstance(self.node[0][attr], list):
                for element in self.node[0][attr]:
                    weight_list.append(weight)
            else:
                weight_list.append(weight)



        for node in self.__iter__():

            a_row = list()

            for attr in attr_config.keys():
                if isinstance(self.node[node][attr], list):
                    for element in self.node[node][attr]:
                        if isinstance(self.node[node][attr], hist):
                            a_row.append(hist_func(self.node[node][attr]))
                        else:
                            a_row.append(element)

                elif isinstance(self.node[node][attr], hist):
                    a_row.append(hist_func(self.node[node][attr]))
                else:
                    a_row.append(self.node[node][attr])

            array_list.append(a_row)


        mul_array = np.array(weight_list, dtype=np.float64)

        #print(list(mul_array))

        fs_array = np.array(array_list, dtype=np.float64)
        fs_array *= mul_array

        return fs_array
#        return array_list




    def hist_to_fs_array(self, name, value=1):
        '''Arange a attribute that is itself a histogram into an array that contains
        one row per node. It serves as data points in feature space for clustering operations.'''

        array_list = list()

        for node in self.__iter__():
            if isinstance(self.node[node][name], hist):
                array_list.append(self.node[node][name](mode='array', normalized=True))
            else:
                raise TypeError("Attribute is of type {type(self.node[node][name])}. Must be hist!")


        fs_array = np.array(array_list, dtype=np.float64)

        fs_array *= value

        return fs_array



    def clustering(self, attr_name, algorithm, fs_array, **cluster_kwargs):
        '''Perform any clustering operation from sklearn.cluster on a given feature space array
        (as returnd by 'get_feature_space_array()' or 'hist_to_fs_array()').
        Return the cluster label of each node as an attribute.'''

        cluster_class = getattr(sklearn.cluster, algorithm)
        cluster_obj = cluster_class(**cluster_kwargs).fit(fs_array)

        for node_ix, label in enumerate(cluster_obj.labels_):
            self.node[node_ix][attr_name] = label
        


#    def kmeans_clustering(self, attr_name, fs_array, k, **cluster_kwargs):
#        '''Perform the KMeans clustering from SKLearn on a geiven feature space array
#        (as returnd by 'get_feature_space_array()' or 'hist_to_fs_array()').
#        Return the cluster label of each node as an attribute.'''
#
#        cluster_obj = KMeans(k, **cluster_kwargs).fit(fs_array)
#
#        for node_ix, label in enumerate(cluster_obj.labels_):
#            self.node[node_ix][attr_name] = label
#
#
#
#    def mean_shift_clustering(self, attr_name, fs_array, **ms_kwargs):
#        '''Perform the MeanShift clustering from SKLearn on a geiven feature space array
#        (as returnd by 'get_feature_space_array()' or 'hist_to_fs_array()').
#        Return the cluster label of each node as an attribute.'''
#
#        meanshift_obj = MeanShift(**ms_kwargs).fit(fs_array)
#
#        for node_ix, label in enumerate(meanshift_obj.labels_):
#            self.node[node_ix][attr_name] = label
#




    def produce_cluster_image(self, attribute, dtype=np.int64):
        '''Render an image (2D numpy array) of cluster labels based
        on a cluster label node attribute.'''

        cluster_img = np.zeros_like(self.seg_img, dtype=dtype)

        for node in self.__iter__():
            for label in set(self.node[node]['labels']):
                mask = self.seg_img == label
                cluster_img[mask] = self.node[node][attribute]

        return cluster_img


    def neighbour_cross_tabulation(self, attribute):
        '''Tabulate the joint distribution of cluster labels
        for all adjacent nodes.'''

        count = Counter()
        for node1, node2, in self.edges():
            combo = tuple(sorted([self.node[node1][attribute], self.node[node2][attribute]]))
            count[combo] += 1
        return count




    @classmethod
    def old_init(cls, seg_img, tex_img, color_image, tex_bins, color_bins, **attr):
        '''Constructor of the first version of this class to ensure backwards compatibility.'''

        new_rag = cls(seg_img, **attr)

        new_rag.add_attribute('tex', tex_img, hist, vbins=tex_bins)
        new_rag.normalize_attribute('tex')

        new_rag.add_attribute('color', color_image, hist, bins=color_bins)
        new_rag.normalize_attribute('color')

        return new_rag








#Simple merging function
def _bow_merge_simple(graph, src, dst):
    '''Function to perform attribute transfer/recalculation
    of two nodes to be merged as part of a sequently merging
    algorithm.'''

    #pixel counter
    graph.node[dst]['pixel_count'] += graph.node[src]['pixel_count']

    #get color values for super pixel
    label_mask = (graph.seg_img == src) | (graph.seg_img == dst)

    for attr, fconfig in graph.attr_func_configs.items():

        masked_image = fconfig.img[label_mask]

        graph.node[dst][attr] = graph.calc_attr_value(data=masked_image,
                                                      func=fconfig.func, **fconfig.kwargs)

        #Normalize according to specs
        if attr in graph.attr_norm_val:
            graph.normalize_attribute(attr, graph.attr_norm_val[attr])
        #else: raise KeyError(f"Attribute '{attr}' has no stored normalization value")
