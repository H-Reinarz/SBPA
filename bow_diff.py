#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 17:45:55 2017

@author: hre070
"""
from collections import namedtuple
from statistics import mean
from histogram import Intersection


#Weighting function: Cumulative difference over bins
def cumu_diff(graph, node1, node2, **kwargs):
    '''Old weighting function. Don't use!'''

    node1 = graph.get_node_data(node1, percentages=True)
    node2 = graph.get_node_data(node2, percentages=True)

    c_diff = 0
    for key, val in node1['tex'].items():
        c_diff += abs(val - node2['tex'][key])
    return c_diff


def _cumu_weight(graph, src, dst, n):
    '''Old weighting function. Don't use!'''
    return {'weight': cumu_diff(graph, dst, n)}


def config_weighting(attr_dict={'tex': (Intersection, 1), 'color': (Intersection, 1) },
                     result_label='weight'):
    '''Closure that recieves a specification of attributes along with associated
    compare function and weight. It produces a weighting function
    with an argument signature compliant with 'merge_hierarchical()'
    that compares graph nodes according to the specification.'''

    weighting_obj = namedtuple('WeightFunc', ['graph', 'merge'])

    def weight_func(graph, node1, node2, neighbor_node, *args, **kwargs):
        '''Weighting function that wraps around attribute compare functions
        and returns a final weight. It gets dynamically defined by the surrounding closure.'''
        #Deep copies
        node1 = graph.deepcopy_node(node1)
        node2 = graph.deepcopy_node(node2)
        neighbor = graph.deepcopy_node(neighbor_node)

        #Iteration
        par_results = []

        for attr, info in attr_dict.items():
            func = info[0]
            factor = info[1]

            if isinstance(node1[attr], list):
                result = mean([func(v[0](), v[1]()) for v in zip(node2[attr], neighbor[attr])])
            else:
                result = func(node2[attr](), neighbor[attr]())

            par_results.append(result*factor)


        final_result = mean(par_results)
        return {result_label: final_result}

    def weight_func_minimal(graph, node1, node2, *args, **kwargs):
        '''Minimal version of 'weight_func()' with different argument signature.'''
        return weight_func(graph, node1=0, node2=node1, neighbor_node=node2, *args, **kwargs)

    return weighting_obj(weight_func_minimal, weight_func)
