#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 16:16:39 2018

@author: hre070
"""

from collections import namedtuple



threshhold = namedtuple('threshhold', ['value', 'operator'])

fs_bundle = namedtuple('bundle', ['graph', 'feature_space', 'metric_dict'])

#IN DEVELOPMENT

class threshhold_stage(object):
    
    def __init__(self, threshhold_dict, **kwargs):
        
        self.threshhold_dict = threshhold_dict
        self.next_stage_true = None
        self.next_stage_false = None
        self.kwargs = kwargs
        
      
    def set_successor_stages(self, true, false):
        self.next_stage_true = true
        self.next_stage_false = false

    def evaluate(self, metric_dict):
        evaluation = False
        
        for metric, thresh in self.threshhold_dict.items():
            if eval(f'{metric_dict[metric]} {thresh.operator} {tresh.value}'):
                evaluation = True
                
        return evaluation
    
    def react_to_true(self, bundle):
        if self.next_stage_true is not None:
            self.next_stage_true.send(bundle)
    
    def react_to_false(self, bundle):
        if self.next_stage_false is not None:
            self.next_stage_false.send(bundle)
        
    def __call__(self):
        while True:
            bundle = yield
            
            assert(isinstance(bundle, fs_bundle))
                    
            if self.evaluate(bundle.metric_dict):
                self.react_to_true(bundle)
            else:
                self.react_to_false(bundle)
                



class cluster_stage(threshhold_stage):
    
    def react_to_true(self, bundle):
        cluster_kwargs = dict(self.kwargs)
        
        for kwarg in ('attribute', 'algorithm', 'fs_spec', 'return_clust_obj'):
            del cluster_kwargs[kwarg]
            
        bundle.graph.clustering(self.kwargs['attribute'], self.kwargs['algorithm'],
                                bundle.feature_space, **cluster_kwargs)



class splitting_stage(threshhold_stage):
    pass
    



def dynamic_clustering_loop(start_bundle, entry_point):
    
    bundle_list = []
    
    bundle_list.append(start_bundle)
    
    for bundle in bundle_list:
        entry_point.send(bundle)
        
        new_bundles = yield
        
        for new_bun in new_bundles:
            bundle_list.append(new_bun)
        

def dynamic_clustering(graph, attr_config, attribute, entry_point):
    
    graph.basic_feature_space_array()
    
    