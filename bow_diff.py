#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 17:45:55 2017

@author: hre070
"""

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


def diff_wrapper(graph, node1, node2, attr_dict, result_label='weight'):
    
    #Code
    
    final_diff = 0
    return {result_label: final_diff}