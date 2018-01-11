#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 16:16:39 2018

@author: hre070
"""

from collections import namedtuple



threshhold = namedtuple('threshhold', ['value', 'operator'])

proc_bundle = namedtuple('ProcessingBundle', ['graph', 'attribute', 'attr_config', 'feature_space', 'metric_dict'])

#IN DEVELOPMENT

class logic_stage(object):
    
    def __init__(self, threshhold_dict=None, **kwargs):
        
        self.threshhold_dict = threshhold_dict
        self.next_stage_true = None
        self.next_stage_false = None
        self.kwargs = kwargs
        
      
    def set_successor_stages(self, true, false):
        self.next_stage_true = true
        self.next_stage_false = false

    def evaluate(self, metric_dict):
        if self.threshhold_dict is not None:
            evaluation = False
            
            for metric, thresh in self.threshhold_dict.items():
                if eval(f'{metric_dict[metric]} {thresh.operator} {tresh.value}'):
                    evaluation = True
                    
            return evaluation
        else:
            return False
        
    def react_to_true(self, bundle):
        if self.next_stage_true is not None:
            self.next_stage_true.send(bundle)
    
    def react_to_false(self, bundle):
        if self.next_stage_false is not None:
            self.next_stage_false.send(bundle)
        
    def __call__(self):
        while True:
            bundle = yield
            
            assert(isinstance(bundle, proc_bundle))
                    
            if self.evaluate(bundle.metric_dict):
                self.react_to_true(bundle)
            else:
                self.react_to_false(bundle)
                



class cluster_stage(logic_stage):
    
    def react_to_true(self, bundle):
        cluster_kwargs = dict(self.kwargs)
        
        for kwarg in ('algorithm'):
            del cluster_kwargs[kwarg]
            
        bundle.graph.clustering(bundle.attribute, self.kwargs['algorithm'],
                                bundle.feature_space, **cluster_kwargs)
        
        if self.next_stage_true is not None:
            self.next_stage_true.send(bundle)




class splitting_stage(logic_stage):
    
    def react_to_false(self, bundle):
        if 'bundle_list' not in self.kwargs or not isinstance(self.kwargs['bundle_list'], list):
            raise ValueError('Object needs a list to append!')
        
        new_fs_list = bundle.graph.attribute_divided_fs_arrays(bundle.attr_config,
                                                               bundle.attribute,
                                                               subset=bundle.feature_space.order)
        
        for fs in new_fs_list:
            metrics = bundle.graph.apply_group_metrics(fs, bundle.metric_config)
            
            new_bundle = proc_bundle(bundle.graph, bundle.attribute, bundle.attr_config, fs, metrics)
            
            self.kwargs['bundle_list'].append(new_bundle)
    
    
    


        

def dynamic_clustering(graph, attr_config, attribute, metric_config, entry_point, hand_back):

    bundle_list = []
    
    hand_back.kwargs['bundle_list'] = bundle_list

    start_fs = graph.basic_feature_space_array(attr_config)
    
    metrics = graph.apply_group_metrics(start_fs, metric_config)
    
    start_bundle = proc_bundle(graph, attribute, start_fs, metrics)
    
    bundle_list.append(start_bundle)
    
    for bundle in bundle_list:
        entry_point.send(bundle)


    
    
    