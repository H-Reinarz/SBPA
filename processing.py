#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 16:16:39 2018

@author: hre070
"""

from collections import namedtuple



threshhold = namedtuple('threshhold', ['value', 'operator'])

proc_bundle = namedtuple('ProcessingBundle', ['graph', 'attribute', 'attr_config',
                                              'metric_config', 'feature_space', 'metric_dict'])

def logic_stage_generator(logic_stage):
    '''Generator definition for LogicStage.'''
    while True:
        bundle = yield
        
        assert(isinstance(bundle, proc_bundle))
                
        if logic_stage.evaluate(bundle.metric_dict):
            logic_stage.react_to_true(bundle)
        else:
            logic_stage.react_to_false(bundle)

    

class LogicStage(object):
    '''Class to recieve a bundle object representing a group
    of notes to process. Decision is made by applying threshholds to a
    set of metrics. It also serves as a base class for more specialized stages
    in the same workflow.'''
    
    def __init__(self, threshhold_dict=None, **kwargs):
        
        self.threshhold_dict = threshhold_dict
        self.next_stage_true = None
        self.next_stage_false = None
        self.kwargs = kwargs
        self.socket = None
        
      
    def set_successor_stages(self, successor_true=None, successor_false=None):
        '''Define the objects to forward the bundle to
        depending on the evaluation.'''
        
        if successor_true is not None:
            assert(isinstance(successor_true, LogicStage))
            
        if successor_false is not None:
            assert(isinstance(successor_false, LogicStage))
            
        self.next_stage_true = successor_true
        self.next_stage_false = successor_false

    def evaluate(self, metric_dict):
        '''Perform the evaluation of a set of metrics
        with the instances threshholds.'''
        if self.threshhold_dict is not None:
            evaluation = False
            
            for metric, thresh in self.threshhold_dict.items():
                if thresh is None: continue
                if eval(f'{metric_dict[metric]} {thresh.operator} {thresh.value}'):
                    evaluation = True
                    
            return evaluation
        else:
            return False
        
    def react_to_true(self, bundle):
        '''Action if evaluate() returns True.'''
        if self.next_stage_true is not None:
            self.next_stage_true.socket.send(bundle)
    
    def react_to_false(self, bundle):
        '''Action if evaluate() returns False.'''
        if self.next_stage_false is not None:
            self.next_stage_false.socket.send(bundle)
        
    def __call__(self):
        '''Starts the generator for the class functionality.'''
        self.socket = logic_stage_generator(self)
        next(self.socket)



class ClusterStage(LogicStage):
    '''Specialized stage that performs a specified clustering
    on the nodes in the recieved bundle.'''
    
    def react_to_true(self, bundle):
        '''Specialized reaction peforming the clustering.'''
        cluster_kwargs = dict(self.kwargs)
        
        
        del cluster_kwargs['algorithm']
        
        print(f'Doing {self.kwargs["algorithm"]} on {bundle.feature_space.label}')
            
        bundle.graph.clustering(bundle.attribute, self.kwargs['algorithm'],
                                bundle.feature_space, **cluster_kwargs)
        
        if self.next_stage_true is not None:
            self.next_stage_true.socket.send(bundle)




class SplittingStage(LogicStage):
    '''Specialized stage to perform the splitting up of a clustered bundle of nodes
    into a new bundle for each cluster.'''
    
    def react_to_false(self, bundle):
        '''Specialized stage to split a bundle.
        Envokes IPAG.attribute_divided_fs_arrays().'''
        if 'bundle_list' not in self.kwargs or not isinstance(self.kwargs['bundle_list'], list):
            raise ValueError('Object needs a list to append!')
        
        print(f'Splitting {bundle.feature_space.label}')
        
        new_fs_list = bundle.graph.attribute_divided_fs_arrays(bundle.attr_config,
                                                               bundle.attribute,
                                                               subset=bundle.feature_space.order)
        
        for fs in new_fs_list:
            metrics = bundle.graph.apply_group_metrics(fs, bundle.metric_config)
            
            new_bundle = proc_bundle(bundle.graph, bundle.attribute, bundle.attr_config,
                                     bundle.metric_config, fs, metrics)
            
            self.kwargs['bundle_list'].append(new_bundle)
    
    
    
class LogicStageDict(dict):
    '''Specialized dictionary to hold instances
    of LogicStage to facilitate their usage.'''
    
    def link_stages(self, stage, successor_true=None, successor_false=None):
        '''Wrapper around LogicState.set_successor_stages()
        to work with keys.'''
        
        if successor_true is not None:
            successor_true = self[successor_true]
            
        if successor_false is not None:
            successor_false = self[successor_false]            
        
        self[stage].set_successor_stages(successor_true, successor_false)

        
    def initiate_stages(self):
        '''Call all contained stage instances and
        prepare them for recieving bundles.'''
        
        for stage in self.values():
            stage()

            
        

def dynamic_clustering(graph, attr_config, attribute, metric_config, entry_point, hand_back):
    '''Function to start a processing workflow involving a network of logic stages.
    It sends the initial bundle to the entry point and recieves the resulting bundles in 
    a list and recursivly feeds them to the entry point.'''

    bundle_list = []
    
    hand_back.kwargs['bundle_list'] = bundle_list

    start_fs = graph.basic_feature_space_array(attr_config)
    
    metrics = graph.apply_group_metrics(start_fs, metric_config)
    
    start_bundle = proc_bundle(graph, attribute, attr_config, metric_config, start_fs, metrics)
    
    bundle_list.append(start_bundle)
    
    for bundle in bundle_list:
        print(f'Sending bundle: {bundle.feature_space.label}')
        entry_point.socket.send(bundle)


    
    
    