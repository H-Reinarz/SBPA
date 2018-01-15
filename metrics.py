from .ipag import IPAG
from collections import namedtuple

metric = namedtuple('metric', ['func', 'kwargs'])

def fs_variance(graph, fs):
    ''' Computes variance for complete feature_space. Variance of every
    feature divided by number of features. '''
    
    if not isinstance(fs, IPAG.feature_space):
        raise TypeError("Must be IPAG.fs_spec!")
    
    variance = 0
    
    for dim in range(0,fs.array.shape[1]):
        variance += fs.array[:,dim].var()
    
    variance /= fs.array.shape[1]
    
    return variance



def cmp_variance(fs, original_var, limit_percent):
    ''' Checks if feature_space variance has reached limit_percent of original
    variance value. If <= original variance function returns True. Else False.
    Example: Is 213 (fs) 30 percent or less (limit_percent) than 250 
    (original_var)? Result: False'''
    
    current_var = fs_variance(fs)
    current_percent = (current_var / original_var) * 100
    if current_percent <= limit_percent:
        return True
    else:
        return False


		
def count_pixel(graph, fs):
    '''Counts pixel of a feature_space'''
    
    if not isinstance(fs, IPAG.feature_space):
        raise TypeError("Must be IPAG.fs_spec!")
    pixel_sum = 0

    for nodes in fs.order:
        pixel_sum += graph.node[nodes]['pixel_count']
            
    return pixel_sum
	

def old_count_pixel(g, fs, pixel_min=0, invert=False):
    '''Counts pixel of a feature_space list. Returns a dictionairy'''
    
    if isinstance(fs, IPAG.feature_space):
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


def count_multi_features(graph, fs, attribute):
    '''Counts features (non continues cluster patches) of multifeature clusters'''
    
    if not isinstance(fs, IPAG.feature_space):
        raise TypeError("Must be IPAG.feature_space!")
    
    attr_check = 0
    for node in fs.order:
        if attribute in graph.node[node]:
            attr_check += 1
    
    if attr_check == len(fs.order):    
        processed = set() # Keep track which node has already been processed
        
        isolateNodes = 0
        
        for node in fs.order:
            if node in processed:
                continue
            # Isolate current node
            count_isolate(graph, node, processed, attribute)
            isolateNodes += 1
        
        return isolateNodes

    elif attr_check == 0:
        return 1
    
    else:
        raise AttributeError(f'Not all nodes have the attribute {attribute}')
        
def count_isolate(graph, node, processed, attribute):
    '''Helper Function of count_multi_features'''
    
    # Node is processed and gets new cluster ID
    processed.add(node)
        
    # Do the same for all neighbors with the same previous cluster
    for neighbour in graph.neighbors(node):
        if (graph.node[node][attribute] == graph.node[neighbour][attribute]) and neighbour not in processed:
            count_isolate(graph, neighbour, processed, attribute)   
            
def count_superpixel(graph, fs):
    return len(fs.order)