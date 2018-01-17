#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 16:16:39 2018

@author: hre070
"""
import sys, traceback
from collections import namedtuple



threshhold = namedtuple('Threshhold', ['operator', 'value'])

proc_bundle = namedtuple('ProcessingBundle', ['graph', 'attribute', 'attr_config',
                                              'metric_config', 'feature_space', 'metric_dict'])

#proc_error = namedtuple('ProcessingErrorEvent', ['type', 'value', 'traceback', 'bundle'])

def _raise(bundle, notify=True):
    raise

class ProcessingErrorEvent():
    '''Class storing all relevant information if an exception
    is raised by a method of a LogicStage during processing.'''
    def __init__(self, exc_type, value, traceback, context, bundle):
        
        self.type = exc_type
        self.value = value
        self.traceback = traceback
        self.context = context
        self.bundle = bundle
        
    
    def __str__(self):
        data = (self.type, self.value, self.context,
                self.bundle.attribute, self.bundle.feature_space.label)
        string = 'Error: {}\nMessage: {}\nContext: {}\nBundle: {} {}'.format(*data)
        
        return string
    
    def print_traceback(self):
        '''Print the instances traceback.'''
        traceback.print_tb(self.traceback, file=sys.stdout)
        

class ExceptionRecorder(list):
    '''Class to record exceptions raised by specialized LogicStage
    instances.'''
    
    def __init__(self, label=None, notify=True):
        super().__init__()
        
        self.label= label
        self.raise_types = {}
        self.notify = notify

    def print_traceback(self, index):
        '''Print full traceback for given entry.'''
        self[index].print_traceback()

    def print_full_report(self):
        '''Print information for all entries.'''
        
        for record in self:
           print(record)
           print('\n')

    def print_full_traceback(self):
        '''Print traceback for all entries.'''
        for record in self:
            record.print_traceback()
            print('\n')
            
    def __call__(self, bundle, context=''):
        '''Wrapper around sys.exc_info() to produce a ProcessingErrorEvent
        namedtuple. Also takes the current bundle at the point the exception was raised.'''
        catched = ProcessingErrorEvent(*sys.exc_info(), context, bundle=bundle)

        self.append(catched)
        
        if self.notify:
            print('Recorded processing error:')            
            print('-'*80)
#            print('Recorded {} while processing bundle: {} {}\nMessage: {}'.format(
#                    catched.type, catched.bundle.attribute, catched.bundle.feature_space.label, catched.value))
            print(catched)
            print('-'*80)            

        if catched.type in self.raise_types:
            raise



def logic_stage_generator(logic_stage):
    '''Generator definition for LogicStage.'''
    assert(isinstance(logic_stage, LogicStage))
    
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
    
    def __init__(self, threshhold_dict=None, context='', **kwargs):
        
        self.threshhold_dict = threshhold_dict
        self.context=''
        self.next_stage_true = None
        self.next_stage_false = None
        self.kwargs = kwargs
        self.socket = None
        self.exception_recorder = _raise
        
      
    def set_successor_stages(self, successor_true=None, successor_false=None):
        '''Define the objects to forward the bundle to
        depending on the evaluation.'''
        
        if successor_true is not None:
            assert(isinstance(successor_true, LogicStage))
            
        if successor_false is not None:
            assert(isinstance(successor_false, LogicStage))
            
        self.next_stage_true = successor_true
        self.next_stage_false = successor_false



    def set_exception_recorder(self, recorder=_raise):
        '''Setter for corresponding ExceptionRecorder instance.'''
        self.exception_recorder = recorder



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
        print("Reacts to True")
        if self.next_stage_true is not None:
            self.next_stage_true.socket.send(bundle)
    
    def react_to_false(self, bundle):
        '''Action if evaluate() returns False.'''
        print("Reacts to False")
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


class IsolateStage(LogicStage):
    '''Specialized stage that splits apart the spatially isolated
    parts of one cluster of nodes.'''

    def react_to_true(self, bundle):
        '''Specialized reaction peforming the isolating.'''
        
        print(f'Isolating {bundle.feature_space.label}')
            
        bundle.graph.isolate(bundle.feature_space, layer=bundle.attribute)
        
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
        
        print(len(new_fs_list))
        
        if len(new_fs_list) == 1:
            
            metrics = bundle.graph.apply_group_metrics(new_fs_list[0], bundle.metric_config)
            print(metrics)
            
            new_bundle = proc_bundle(bundle.graph, bundle.attribute, bundle.attr_config,
                                     bundle.metric_config, new_fs_list[0], metrics)
            
            self.kwargs['bundle_list'].append(new_bundle)
            
            self.react_to_true(new_bundle)
        
        else:
            for fs in new_fs_list:
                metrics = bundle.graph.apply_group_metrics(fs, bundle.metric_config)
                print(metrics)
                
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


    
#Testing code    
if __name__ == '__main__':
    
    
    mock_fs = namedtuple('FS', ['label'])
    
    mock_bundle = proc_bundle('Graph', 'cluster', {'color':0.5}, {'size':'func'},
                              mock_fs(['0','1','0']), {'size':42})
    
    er = ExceptionRecorder()
    
    try:
        raise ValueError(42)
    except:
        er(mock_bundle)
    
    
    try:
        raise TimeoutError('bad luck')
    except:
        er(mock_bundle)

