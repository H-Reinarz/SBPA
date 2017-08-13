#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 14:25:44 2017

@author: hre070
"""

#Imports
from skimage.future import graph
import networkx as nx
from itertools import repeat
import numpy as  np

from bow_diff import cumu_diff


#Subclass of RAG specified for BOW classification

class BOW_RAG(graph.RAG):
    
    def __init__(self, seg_img, word_img, bins, **attr):
        
        #Call the RAG constructor
        super().__init__(label_image=seg_img, connectivity=2, data=None, **attr)
        
        #Build tuple of BOW-dictionaries
        bow_tuple = (dict(zip(bins,repeat(0, len(bins)))) for node in self.nodes_iter())
        
        #Set BOW attribute to nodes by attaching node id's to bow dicts
        nx.set_node_attributes(self, "BOW", dict(zip(self.nodes_iter(), bow_tuple)))
        
        #Set pixel counting attribute to each node to count the pixels per segment
        nx.set_node_attributes(self, "PIXELS", dict(zip(self.nodes_iter(), repeat(0, self.number_of_nodes()))))
        
        #Populate the node attributes with data
        for a, b in np.nditer([seg_img, word_img]):
            
            #Pixel count incrementation
            self.node[int(a)]["PIXELS"] += 1
            
            #BOW attribute individual bin incrementation
            self.node[int(a)]["BOW"][int(b)] += 1
            
        #Turn count values into percentages
        for node, data in self.nodes_iter(data=True):
            for key, value in data["BOW"].items():
                data["BOW"][key] = round((value/data["PIXELS"])*100, 3)



    def calc_edge_weights(self, weight_func = cumu_diff, attr_label="WEIGHT", **kwargs):
        
        #Iterate over edges and calling weight_func on the nodes
        for n1, n2, d in self.edges_iter(data=True):
            d[attr_label] = weight_func(self.node[n1], self.node[n2], **kwargs)
            
            
