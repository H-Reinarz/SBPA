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
        
        
        processed = set() # Keep track which node has already been processed
        
    
        for node in fs.order:
            if node in processed:
                continue
            
            neighbour_list = [node]
            neighbour_set = set(neighbour_list)
            for neighbour in neighbour_list:
                equal_neighbours = graph.get_equal_neighbors(neighbour, attribute)
                for en in equal_neighbours:
                    if en not in neighbour_set:
                        neighbour_list.append(en) #+= list(neighbour_set.difference(equal_neighbours))
                neighbour_set.update(neighbour_list)
                
            isolateNodes += 1

            processed.update(neighbour_list)
        
        return isolateNodes

    elif attr_check == 0:
        return 1
    
    else:
        raise AttributeError(f'Not all nodes have the attribute {attribute}')
        
def get_equal_neighbors(graph, node, processed, attribute):
    '''Helper Function of count_multi_features. Returns neighbors of a node 
    that have the same label.'''
    
    # Do the same for all neighbors with the same previous cluster
    filter_n = lambda n: graph.node[node][attribute] == graph.node[n][attribute]
    
    return filter(filter_n, graph.neighbors(node))  
            
def count_superpixel(graph, fs):
    return len(fs.order)