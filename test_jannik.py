# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 10:38:45 2017

@author: Jannik
"""

import sys
sys.path.append("D:/janni/Documents/Geographie/Masterarbeit/src/github")
sys.path.append("D:/janni/Documents/Geographie/Masterarbeit/src/github/_LBP")

from skimage import io
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage import data, segmentation
from skimage.future.graph import RAG
import networkx as nx
import bow_rag
import bow_diff
import numpy as np

from skimage.color import rgb2gray
from matplotlib import pyplot as plt



def SingleFeature(rag, attr_name_in, attr_name_out):
    
    rag_tmp = rag.copy()
    
    # Keep track which node has already been processed
    for node in rag:
        rag_tmp.node[node]['processed'] = False
    
    # New Cluster ID        
    clusters = 0
    
    for node in rag:
        if rag_tmp.node[node]['processed']:
            continue
        
        # Isolate current node
        Isolate(rag, node, clusters, rag_tmp, attr_name_in, attr_name_out)
        clusters += 1

def Isolate(rag, node, clusters, rag_tmp, attr_name_in, attr_name_out):
    
    # Node is processed and gets new cluster ID
    rag_tmp.node[node]['processed'] = True
    rag.node[node][attr_name_out] = clusters
    
    # Do the same for all neighbors with the same previous cluster
    for neighbour in rag.neighbors_iter(node):
        if (rag_tmp.node[node][attr_name_in] == rag_tmp.node[neighbour][attr_name_in]) and rag_tmp.node[neighbour]['processed'] == False:
            
            Isolate(rag, neighbour, clusters, rag_tmp, attr_name_in, attr_name_out)   


def Absorb(rag, size, attr_name_in, attr_name_out = "cluster_new"):
    
    rag_tmp = rag.copy() # Copy rag for adding tmp attributes
    
    clusterSet = set() # unique cluster set
    
    for node in rag:
        rag_tmp.node[node]['absorbed'] = False # tmp control attribute
        rag.node[node][attr_name_out] = rag.node[node][attr_name_in] # new cluster ID = current cluster ID for start
        clusterSet.add(rag.node[node][attr_name_in]) # create unique cluster set
    
    clusterDict = {c: list() for c in clusterSet} # cluster set to cluster dict
    
    for node in rag:
        # add all nodes to key entry of their current cluster
        # to keep track which node is in which cluster
        clusterDict[rag.node[node][attr_name_in]].append(node) 
    
    for key, value in clusterDict.items():
        pixelCount = 0
        neighbors = list()
        
        for node in value:
            # Go trough all clusters
            # Get neighbors of cluster
            # Get pixelcount of cluster
            neighbors = neighbors + rag.neighbors(node)
            pixelCount += rag.node[node]['pixel_count']
        
        neighbors = set(neighbors)
        
        if len(value) > 1:
            for node in value:
                neighbors.remove(node) # Remove cluster own nodes from neighbor set
        
        
        if pixelCount <= size:
            neighborDict = {n: 0 for n in clusterSet}
            # Count neighbors by cluster ID
            for n in neighbors:
                neighborDict[rag.node[n][attr_name_in]] += 1
            # Cluster ID with most counts will absorb the current cluster
            newCluster = max(neighborDict.keys(), key=(lambda key: neighborDict[key]))
            
            # absorb
            for node in value:
                rag.node[node][attr_name_out] = newCluster
            

            
        
        
        

        
    
    
    
        
        
    


image = io.imread("D:/janni/Documents/Geographie/Masterarbeit/Data/ra_neu/ra2_small.jpg")
im_gray = rgb2gray(image)
segments_slic = slic(image, n_segments=800, compactness=15, sigma=1)
Graph = bow_rag.BOW_RAG(segments_slic)

Graph.add_attribute('color', image, np.mean)
Graph.normalize_attribute('color', value=255)
Graph.add_attribute('var', im_gray, np.var)
fs_attrs = {'color':1, 'var':2}
fs_array = Graph.get_feature_space_array(fs_attrs)
Graph.clustering("cluster", "KMeans", fs_array, n_clusters = 5)

cluster_img1 = Graph.produce_cluster_image('cluster')
from skimage import color

import time
time1 = time.time()
SingleFeature(Graph, "cluster", "clusterbob")
time2 = time.time()
print ((time2-time1)*1000.0, ", ms")

print(Graph.node[0])
cluster_img2 = Graph.produce_cluster_image('clusterbob')


from visual import plot_sp_labels

out = color.label2rgb(cluster_img1, image, kind='avg')
out = segmentation.mark_boundaries(out, cluster_img1, (0, 0, 0))

f, ax = plt.subplots(figsize=(15, 15))

ax.imshow(out)
plot_sp_labels(ax, cluster_img1, 13, color="white")

 

out = color.label2rgb(cluster_img2, image, kind='avg')
out = segmentation.mark_boundaries(out, cluster_img2, (0, 0, 0))

f, ax = plt.subplots(figsize=(15, 15))

ax.imshow(out)
plot_sp_labels(ax, cluster_img2, 13, color="white")

 



Absorb(Graph, 5000, "clusterbob", "clusterbob")
cluster_img3 = Graph.produce_cluster_image('clusterbob')

out = color.label2rgb(cluster_img3, image, kind='avg')
out = segmentation.mark_boundaries(out, cluster_img3, (0, 0, 0))

f, ax = plt.subplots(figsize=(15, 15))

ax.imshow(out)
plot_sp_labels(ax, cluster_img3, 13, color="white")
        
        
        
        
        
        
""" 

    for node in rag:
        if rag.node[node]['x']:
            continue
        
        neighbors = rag.neighbours(node)
        
        while not bRag.node[node]['x']:
            
            for neighbour in neighbors:
                if rag.node[node]['cluster'] == rag.node[neighbour]['cluster']:
                    dst_nbrs = set(rag.neighbors(rag.node[neighbour]['cluster']))
                    neighbors = (neighbors | dst_nbrs) - set([neighbors, dst])    
            
        
        for neighbour in bRag.neighbors_iter(node):
            if bRag.node[node]['cluster'] == bRag.node[neighbour]['cluster']:
                
"""  