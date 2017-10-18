#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mon Aug  7 14:25:44 2017

@author: hre070
'''

#Imports
from skimage.future import graph
#import networkx as nx
#from itertools import repeat
import statistics as stats
import numpy as  np
import copy
from collections import namedtuple

from bow_container import hist

#Subclass of RAG specified for BOW classification

class BOW_RAG(graph.RAG):
    
    config = namedtuple('AttributeConfig', ['img', 'func', 'kwargs'])
    
    def __init__(self, seg_img, **attr):
        
        #Call the RAG constructor
        super().__init__(label_image=seg_img, connectivity=1, data=None, **attr)
        
        #Store seg_img as attribute
        self.seg_img = seg_img
        
        self.attr_func_configs = {}
                        
        #Init edge weight statistics
        self.edge_weight_stats = {}
        

        #Set indipendent node attributes
        for n in self.__iter__():
            #get color values for super pixel
            label_mask = self.seg_img == n            

            #Assign attributes to node
            self.node[n].update({'labels': [n],
                              'pixel_count': seg_img[label_mask].size})

    
    def calc_attr_value(self, *, data, func, **kwargs):
            
        #account for multi channels: one hist per channel
        if len(data.shape) == 2:
            result = [func(data[:,dim], **kwargs) for dim in range(data.shape[1])]
        else:
            result = func(data, **kwargs)
            
        return result
        


    def add_attribute(self, name, image, function, input_as, **func_kwargs):
        
        self.node_attr_funcs[name] = BOW_RAG.config(image, function, func_kwargs)
        
        #Set node attributes
        for n in self.__iter__():
            #get color values for super pixel
            label_mask = self.seg_img == n
            masked_image = image[label_mask]
            
            attr_value = self.calc_attr_value(data=masked_image, func=function, **func_kwargs)
            
        #Assign attributes to node
        self.node[n].update({name:attr_value})




    
    def normalize_attribute(self, attribute, value=None):
        
        for n in self.__iter__():
            if isinstance(self.node[n][attribute], list):
                for element in list:
                    if isinstance(element, hist): element.normalize(self.node[n]['pixel_count'])
                    else: element/value
            
            else:
                if isinstance(self.node[n][attribute], hist): element.normalize(self.node[n]['pixel_count'])
                else: self.node[n][attribute]/value
               


    
    def delete_attributes(self, attribute):
        for n in self.__iter__:
            del self.node[attribute]

    
    def deepcopy_node(self, node):       
        #create mutable copy of the node for calculation
        return copy.deepcopy(self.node[node])
        


    def calc_edge_weights(self, weight_func):
        
        #Iterate over edges and calling weight_func on the nodes
        for n1, n2, d in self.edges_iter(data=True):
            d.update(weight_func(self, n1, n2))
            
     
    def get_edge_weight_list(self, attr_label='weight'):
        return sorted(list(data[attr_label] for n1,n2,data in self.edges(data=True)))
        
        
    def calc_edge_weight_stats(self, attr_label='weight'):
        weight_list = self.get_edge_weight_list(attr_label)
        
        self.edge_weight_stats['min'] = min(weight_list)
        self.edge_weight_stats['max'] = max(weight_list)
        self.edge_weight_stats['mean'] = stats.mean(weight_list)
        self.edge_weight_stats['median'] = stats.median(weight_list)
        self.edge_weight_stats['stdev'] = stats.stdev(weight_list)

    
    def get_edge_weight_percentile(self, p, attr_label='weight', as_threshhold=False):
        weight_list = self.get_edge_weight_list(attr_label)
        
        index = round(len(weight_list)*(p/100))
        
        if as_threshhold:
           result = (weight_list[index] +  weight_list[index+1]) /2
           return result
        else:
            return weight_list[index]
        
        
        
#Simple merging function
def _bow_merge_simple(graph, src, dst):
    
    #pixel counter
    graph.node[dst]['pixel_count'] += graph.node[src]['pixel_count']
    
    #get color values for super pixel
    label_mask = (graph.seg_img == src) | (graph.seg_img == dst)
    
    for attr, fconfig in graph.attr_func_configs.items():

        masked_image = fconfig.data[label_mask]
        
        graph.node[dst][attr] = graph.calc_attr_value(data=masked_image, func=fconfig.func, **fconfig.kwargs)
        
        
    