import bow_rag
import numpy as np
import networkx as nx

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
		
def CountPixel(g, fs, pixel_min=0, invert=False):
    if isinstance(fs, bow_rag.BOW_RAG.fs_spec):
        fs = [fs]
    clusterDict = {} # unique cluster set
    for _fs in fs:
        clusterDict[str.join(_fs.label)] = 0
        for nodes in _fs.order:
            clusterDict[str.join(_fs.label)] += g.node[nodes]['pixel_count']
            
    if not invert:
        clusterDict = {k: v for k, v in clusterDict.items() if v >= pixel_min}
    else:
        clusterDict = {k: v for k, v in clusterDict.items() if v < pixel_min}
    return clusterDict
	
def CountMultiFeatures(rag, fs, layer):
    '''Counts features (non continues cluster patches) of multifeature clusters'''
    
    if not isinstance(fs, bow_rag.BOW_RAG.fs_spec):
        raise TypeError("Must be BOW_RAG.fs_spec!")
        
    processed = set() # Keep track which node has already been processed
    
    isolateNodes = 0
    
    for node in fs.order:
        if node in processed:
            continue
        # Isolate current node
        CountIsolate(rag, node, processed, layer)
        isolateNodes += 1
    
    return isolateNodes


def CountIsolate(rag, node, processed, layer):
    # Node is processed and gets new cluster ID
    processed.add(node)
        
    # Do the same for all neighbors with the same previous cluster
    for neighbour in rag.neighbors_iter(node):
        if (rag.node[node][layer] == rag.node[neighbour][layer]) and neighbour not in processed:
            Isolate(rag, neighbour, processed, layer)   