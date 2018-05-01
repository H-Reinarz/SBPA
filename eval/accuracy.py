#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 11:26:03 2018

@author: hre070
"""

import numpy as np
from statistics import mean


from .boundary import _get_length


def single_label_jaccard(reference, label, segmentation, only_max=True):
    
    assert reference.shape == segmentation.shape
    
    mask = reference == label
    
    values = np.unique(segmentation[mask])
    
    jaccard_dict = {}
    
    for value in values:
        value_mask = segmentation == value
        
        intersection = np.logical_and(mask, value_mask)
    
        union = np.logical_or(mask, value_mask)

        jaccard_dict[value] = intersection.sum() / float(union.sum())
    
    if only_max:
        return max(jaccard_dict.values())
    else:
        return jaccard_dict






def get_shape_complexity(reference, label):
    mask = reference == label
    
    boundary_length = _get_length(mask.astype(np.int32))

    boundary_circle = (boundary_length/(2*np.pi))**2 * np.pi

#    print(label)
#    print('Boundary:', boundary_length)    
#    print('EQATION:', mask.sum(), '/', boundary_circle, '=', (mask.sum()/boundary_circle))
#    print('='*40)
    
    return (mask.sum()/boundary_circle)




def get_accuracy_metrics(reference, segmentation):
    
    return {label: (get_shape_complexity(reference, label), single_label_jaccard(reference, label, segmentation)) \
            for label in np.unique(reference)}






def get_accuracy_index(reference, segmentation, mean_func=mean):
    
    assert reference.shape == segmentation.shape
    
    labels = np.unique(reference)
    
    complexity_dict = {label: 2-get_shape_complexity(reference, label) for label in labels}

    av_complexity = mean_func(list(complexity_dict.values()))

    perf_dict = {label: av_complexity + abs(abs(cx) - abs(av_complexity)) for label, cx in complexity_dict.items()}
    
    results = {label: (perf * single_label_jaccard(reference, label, segmentation))-1 for label, perf in perf_dict.items()}

#    print('Boundary complexities:',complexity_dict)
#    print('mean complexity:', av_complexity)
#    print('perf dict: ',perf_dict)
    
    return results


if __name__ == '__main__':
    ref = np.array([[2,2,2,2,2,2,2,2,2,2],
                    [2,0,0,0,0,0,0,0,0,2],
                    [2,0,1,1,1,1,1,0,0,2],
                    [2,0,1,1,1,1,1,0,0,2],
                    [2,0,1,1,1,1,1,0,0,2],
                    [2,0,1,1,1,1,1,0,0,2],
                    [2,0,0,0,0,0,0,0,0,2],
                    [2,0,0,0,0,3,3,0,0,2],
                    [2,0,0,0,0,3,3,0,0,2],
                    [2,2,2,2,2,2,2,2,2,2]])
    
    #ref = np.array([[0,0,0,0,0,0,0,0,0,0],
    #                [0,0,0,0,0,0,0,0,0,0],
    #                [0,0,0,0,0,0,0,0,0,0],
    #                [0,0,0,1,1,1,1,0,0,0],
    #                [0,0,0,1,1,1,1,0,0,0],
    #                [0,0,0,1,1,1,1,0,0,0],
    #                [0,0,0,0,0,0,0,0,0,0],
    #                [0,0,0,0,0,0,0,0,0,0],
    #                [0,0,0,0,0,0,0,0,0,0],
    #                [0,0,0,0,0,0,0,0,0,0]])
    
        
        
        
    seg = np.array([[0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,1,1,1,1,1,0,0,0],
                    [0,0,1,1,1,1,1,0,0,0],
                    [0,0,1,1,1,1,1,0,0,0],
                    [0,0,1,1,1,1,1,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0]])
    
    
    
        
    #seg = np.array([[0,0,0,0,0,0,0,0,0,0],
    #                [0,0,0,0,0,0,0,0,0,0],
    #                [0,0,1,1,1,1,1,0,0,0],
    #                [0,0,1,1,1,1,1,0,0,0],
    #                [0,0,1,1,2,2,2,0,0,0],
    #                [0,0,1,1,2,2,2,0,0,0],
    #                [0,0,0,0,0,0,0,0,0,0],
    #                [0,0,0,0,0,0,0,0,0,0],
    #                [0,0,0,0,0,0,0,0,0,0],
    #                [0,0,0,0,0,0,0,0,0,0]])
    
        
    #seg = np.array([[0,0,0,0,0,0,0,0,0,0],
    #                [0,0,0,0,0,0,0,0,0,0],
    #                [0,0,0,0,0,0,0,0,0,0],
    #                [0,0,0,0,0,0,0,0,0,0],
    #                [0,0,0,0,0,0,0,0,0,0],
    #                [0,0,0,0,0,0,0,0,0,0],
    #                [0,0,0,0,0,0,0,0,0,0],
    #                [0,0,0,0,0,0,0,0,0,0],
    #                [0,0,0,0,0,0,0,0,0,0],
    #                [0,0,0,0,0,0,0,0,0,0]])
    #    
        
    print(single_label_jaccard(ref, 1, seg, only_max=False))
    
    print(get_accuracy(ref, seg))
    
    
    #mask = seg == 0
    #
    #int_mask = mask.astype(np.int32)
    #
    #print('Cython:',_get_length(int_mask))