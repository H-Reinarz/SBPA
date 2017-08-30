#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 17:45:55 2017

@author: hre070
"""
from stats import mean
from histogram import Intersection


#Weighting function: Cumulative difference over bins   
def cumu_diff(graph, node1, node2, **kwargs):
    
    n1 = graph.get_node_data(node1, percentages=True)
    n2 = graph.get_node_data(node2, percentages=True)
     
    c_diff = 0
    for k,v in n1['tex'].items():
        c_diff += abs(v-n2['tex'][k])
    return c_diff


def _cumu_weight(graph, src, dst, n):
    
    return {'weight': cumu_diff(graph, dst, n)}


def weight_wrapper(graph, node1, node2, attr_dict={'tex': (Intersection, 1), 'color': (Intersection, 1) }, result_label='weight'):
    
    #Deep copies
    n1 = graph.deepcopy_node(node1)
    n2 = graph.deepcopy_node(node2)
    
    #Iteration
    par_results = []
    
    for attr, info in attr_dict.items():
        func = info[0]
        factor = info[1]
        
        if isinstance(n1[attr], list):
            result = mean([func(v[0](), v[1]()) for v in zip(n1[attr], n2[attr])])
        else:
            result = func(n1[attr](), n2[attr]())
            
        par_results.append(result*factor)
        
        
    
    
    final_result = mean(par_results)
    return {result_label: final_result}