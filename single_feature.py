# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 10:38:45 2017

@author: Jannik
"""
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