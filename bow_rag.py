#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 14:25:44 2017

@author: hre070
"""

#Imports
from skimage.future import graph
#import networkx as nx
from itertools import repeat
import statistics as stats
import numpy as  np
import copy

from bow_diff import cumu_diff


#Subclass of RAG specified for BOW classification

class BOW_RAG(graph.RAG):
    
    def __init__(self, seg_img, word_img, bins, **attr):
        
        #Call the RAG constructor
        super().__init__(label_image=seg_img, connectivity=2, data=None, **attr)
                
        #Set node attributes
        for n in self.__iter__():
            self.node[n].update({'labels': [n],
                              'pixel count': 0,
                              'bow': dict(zip(bins,repeat(0, len(bins))))})

        #Populate the node attributes with data
        for a, b in np.nditer([seg_img, word_img]):
            
            #Pixel count incrementation
            self.node[int(a)]["pixel count"] += 1
            
            #BOW attribute individual bin incrementation
            self.node[int(a)]["bow"][int(b)] += 1
            
        #Init edge weight statistics
        self.edge_weight_stats = {}

    
    def get_node_data(self, node, percentages=False):
        
        #create mutable copy of the node for calculation
        node_copy = copy.deepcopy(self.node[node])
        
        if percentages:
            for key, value in node_copy["bow"].items():
                node_copy["bow"][key] = round((value/node_copy["pixel count"])*100, 3)
            
            #return with counts transformed into percentages
            return node_copy
        
        else:
            #return raw node attributes
            return node_copy
        


    def calc_edge_weights(self, weight_func = cumu_diff, attr_label="weight", **kwargs):
        
        #Iterate over edges and calling weight_func on the nodes
        for n1, n2, d in self.edges_iter(data=True):
            d[attr_label] = weight_func(self, n1, n2, **kwargs)
            
     
    def get_edge_weight_list(self, attr_label="weight"):
        return sorted(list(data[attr_label] for n1,n2,data in self.edges(data=True)))
        
        
    def calc_edge_weight_stats(self, attr_label="weight"):
        weight_list = self.get_edge_weight_list(attr_label)
        
        self.edge_weight_stats['min'] = min(weight_list)
        self.edge_weight_stats['max'] = max(weight_list)
        self.edge_weight_stats['mean'] = stats.mean(weight_list)
        self.edge_weight_stats['median'] = stats.median(weight_list)
        self.edge_weight_stats['stdev'] = stats.stdev(weight_list)

    
    def get_edge_weight_percentile(self, p, attr_label="weight", as_threshhold=False):
        weight_list = self.get_edge_weight_list(attr_label)
        
        index = round(len(weight_list)*(p/100))
        
        if as_threshhold:
           result = (weight_list[index] +  weight_list[index+1]) /2
           return result
        else:
            return weight_list[index]
        
        
        
#Simple merging function
def _bow_merge_simple(graph, src, dst):
    
    graph.node[dst]['pixel count'] += graph.node[src]['pixel count']
    
    for key, val in graph.node[src]['bow'].items():
        graph.node[dst]['bow'][key] += val
