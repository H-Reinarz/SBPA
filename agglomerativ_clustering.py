# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 11:14:34 2017

@author: Jannik
"""
import numpy as np
import networkx as nx
import bow_rag
from sklearn.cluster import AgglomerativeClustering


def AgglCluster(g, attr_name, fs_spec, n_clusters=2, pixel_min=-1, superpixel_min=2, original_variance = None, limit_percent=30):
    if isinstance(fs_spec, bow_rag.BOW_RAG.fs_spec):
        connectivity = nx.adjacency_matrix(g, weight=None)
        g.clustering(attr_name, 'AgglomerativeClustering', fs_spec, n_clusters=n_clusters, linkage="ward", connectivity=connectivity)
    
    elif isinstance(fs_spec, list):
        for node in g.__iter__():
                g.node[node][attr_name] = None
        for fs in fs_spec:
            if not isinstance(fs, bow_rag.BOW_RAG.fs_spec):
                raise TypeError("Must be BOW_RAG.fs_spec!")
            
            if len(fs.order) < superpixel_min:
                for n in fs.order:
                        g.node[n][attr_name] = str(fs.label)
                continue
            
            subset = g.subgraph(nbunch=list(fs[1]))
            
            
            # OLD
            pixel_count = 0
            
            if pixel_min >= 0:
                for n in subset:
                    pixel_count += subset.node[n]['pixel_count']
                if pixel_count <= pixel_min:
                    #print("In FS ", fs.label, " are ", pixel_count, " Pixel. It gets Name ", fs.label)
                    for n in subset:
                        g.node[n][attr_name] = str(fs.label)# + str(4)
                    continue
            
            if original_variance is not None:
                if Cmp_Variance(fs, original_variance, limit_percent):
                    for n in subset:
                        g.node[n][attr_name] = str(fs.label)
                    continue
            
            connectivity = nx.adjacency_matrix(subset, weight=None)
            
            cluster_obj = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward',
                   connectivity=connectivity).fit(fs.array)  
            
            
            # NEW
#            noGrow = False
#    
#            if pixel_min >= 0:
#                clusterDict = {}
#                for label in cluster_obj.labels_:
#                    clusterDict[label] = 0
#                for node, label in zip(fs.order, cluster_obj.labels_):
#                    clusterDict[label] += subset.node[node]['pixel_count']
#                print(clusterDict)
#                for key, value in clusterDict.items():
#                    if clusterDict[key] <= pixel_min:
#                        for n in subset:
#                            g.node[n][attr_name] = str(fs.label)
#                        noGrow = True
#                        break
#            
#            if noGrow:
#                continue
                             
            for node, label in zip(fs.order, cluster_obj.labels_):
                g.node[node][attr_name] = str(fs.label) + str(label)


def AgglCluster_Cascade(g, fs_attr, attr_name, n_cascade=2, automatic=False, pixel_min=-1, superpixel_min= 2, variance = True, limit_percent=30):
    runs = 0
    cascade = True
    fs1 = g.basic_feature_space_array(fs_attr)
    if variance:
        original_variance = Fs_Variance(fs1)
    else:
        original_variance = None
    
    if not automatic:
        for run in range(0, n_cascade):
            AgglCluster(g, attr_name+str(run), fs1, 2, pixel_min, superpixel_min, original_variance, limit_percent)        
            fs1 = g.attribute_divided_fs_arrays(fs_attr, attr_name+str(run))
        runs = n_cascade
    else:
        while cascade:            
            AgglCluster(g, attr_name+str(runs), fs1, 2, pixel_min, superpixel_min, original_variance, limit_percent)        
            fs1 = g.attribute_divided_fs_arrays(fs_attr, attr_name+str(runs))
            if runs > 0:
                for n in g:
                    if g.node[n][attr_name+str(runs-1)] != g.node[n][attr_name+str(runs)]:
                        cascade = True
                        break
                    else:
                        cascade = False
            runs += 1
    return runs


def Fs_Variance(fs):
    ''' Computes variance for complete feature_space. Variance of every
    feature divided by number of features. '''
    
    if not isinstance(fs, bow_rag.BOW_RAG.fs_spec):
        raise TypeError("Must be BOW_RAG.fs_spec!")
    
    variance = 0
    
    for dim in range(0,fs.array.shape[1]):
        variance += fs.array[:,dim].var()
    
    variance /= fs.array.shape[1]
    
    return variance

def Cmp_Variance(fs, original_var, limit_percent):
    ''' Checks if feature_space variance has reached limit_percent of original
    variance value. If <= original variance function returns True. Else False.
    Example: Is 213 (fs) 30 percent or less (limit_percent) than 250 
    (original_var)? Result: False'''
    
    current_var = Fs_Variance(fs)
    current_percent = (current_var / original_var) * 100
    if current_percent <= limit_percent:
        return True
    else:
        return False