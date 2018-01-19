#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 16:16:39 2018

@author: hre070
"""
import sys, traceback
from collections import namedtuple, deque
from itertools import chain


threshhold = namedtuple('Threshhold', ['operator', 'value'])

proc_bundle = namedtuple('ProcessingBundle', ['graph', 'attribute', 'attr_config',
                                              'metric_config', 'feature_space', 'metric_dict'])
def _format_b(b):
    length = len(b.feature_space.label)
    if length != 0:
        layer = b.feature_space.label[-1]
    else:
        layer = '<empty>'
    
    string = '{}[layer:{} label:{}]'.format(b.attribute, length, layer)
    return string

#proc_error = namedtuple('ProcessingErrorEvent', ['type', 'value', 'traceback', 'bundle'])

def _raise(bundle=None, context=''):
    if bundle is not None:
        data = (context, bundle.attribute, bundle.feature_space.label)
        print('Context: {}\nBundle: {} {}'.format(*data))
    else:
        print('Context: '+context)
        
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
        if self.bundle is not None:
            data = (self.type, self.value, self.context, _format_b(self.bundle))
            string = 'Error: {}\nMessage: {}\nContext: {}\nBundle: {}'.format(*data)
        else:
            data = (self.type, self.value, self.context)
            string = 'Error: {}\nMessage: {}\nContext: {}'.format(*data)
                    
        return string
    
    def print_traceback(self):
        '''Print the instances traceback.'''
        print(self.__str__()+'\nTraceback:')
        traceback.print_tb(self.traceback, file=sys.stdout)
        

class ExceptionRecorder(list):
    '''Class to record exceptions raised by specialized LogicStage
    instances.'''
    
    def __init__(self, label, notify=True):
        super().__init__()
        
        self.label = label
        self.raise_types = set()
        self.notify = notify
        self.raise_none_bundle_errors=True

    def print_traceback(self, index):
        '''Print full traceback for given entry.'''
        self[index].print_traceback()

    def print_full_report(self):
        '''Print information for all entries.'''
        print('Full report of '+self.label)
        print('-'*80)
        for record in self:
           print(record)
           print('\n')
        print('-'*80)
        
    def print_full_traceback(self):
        '''Print traceback for all entries.'''
        print('Full traceback of '+self.label)
        print('-'*80)
        for record in self:
            record.print_traceback()
            print('\n')
        print('-'*80)

            
    def __call__(self, bundle=None, context=''):
        '''Wrapper around sys.exc_info() to produce a ProcessingErrorEvent
        namedtuple. Also takes the current bundle at the point the exception was raised.'''
        catched = ProcessingErrorEvent(*sys.exc_info(), context, bundle=bundle)

        self.append(catched)
        
        if self.notify:
            print('Recorded processing error:')            
            print('-'*80)
            print(catched)
            print('-'*80)            

        if catched.type in self.raise_types:
            raise

        if bundle is None and self.raise_none_bundle_errors:
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
    
    def __init__(self, final_stage, threshhold_dict=None, descr=None, **kwargs):
        
        self.threshhold_dict = threshhold_dict
        
        if descr is not None:
            self.descr = descr
        else:
            self.descr = self.__repr__()
            
        self.next_stage_true = None
        self.next_stage_false = None
        self.queue_true = deque()
        self.queue_false = deque()
        self.final_stage = final_stage
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
        try:
            if self.threshhold_dict is not None:
                evaluation = False
                
                for metric, thresh in self.threshhold_dict.items():
                    if thresh is None: continue
                    if eval(f'{metric_dict[metric]} {thresh.operator} {thresh.value}'):
                        evaluation = True
                        
                return evaluation
            else:
                return True
        except:
            self.exception_recorder(bundle=None, context='Error in evaluate() of '+self.descr)

    def react_to_true(self, bundle):
        '''Action if evaluate() returns True.'''
        print(self.descr + " reacts to True")
        try:
            if self.next_stage_true is not None:
                if self.final_stage:
                    self.next_stage_true.queue_true.append((True, bundle, self.descr))
                else:
                    self.next_stage_true.socket.send(bundle)
        except:
            self.exception_recorder(bundle, context='Error in react_to_true() of '+self.descr)

    def react_to_false(self, bundle):
        '''Action if evaluate() returns False.'''
        print(self.descr + " reacts to False")
        try:
            if self.next_stage_false is not None:
                if self.final_stage:
                    self.next_stage_false.queue_false.append((False, bundle, self.descr))
                else:
                    self.next_stage_false.socket.send(bundle)
        except:
            self.exception_recorder(bundle, context='Error in react_to_false() of '+self.descr)
            
    def __call__(self):
        '''Starts the generator for the class functionality.'''
        try:
            self.socket = logic_stage_generator(self)
            next(self.socket)
        except:
            self.exception_recorder(context='Error while initiating '+self.descr)



class ClusterStage(LogicStage):
    '''Specialized stage that performs a specified clustering
    on the nodes in the recieved bundle.'''

    
    def react_to_true(self, bundle):
        '''Specialized reaction peforming the clustering.'''
        cluster_kwargs = dict(self.kwargs)
        
        
        del cluster_kwargs['algorithm']
        
        print(f'Doing {self.kwargs["algorithm"]} on bundle '+ _format_b(bundle))
        
        try:
            bundle.graph.clustering(bundle.attribute, self.kwargs['algorithm'],
                                    bundle.feature_space, **cluster_kwargs)
        except:
            self.exception_recorder(bundle, context='Error while clustering in '+self.descr)

        try:
            if self.next_stage_true is not None:
                self.next_stage_true.socket.send(bundle)
        except:
            self.exception_recorder(bundle,
                                    context='Sending of bundle after clustering failed in '+self.descr)


class IsolateStage(LogicStage):
    '''Specialized stage that splits apart the spatially isolated
    parts of one cluster of nodes.'''

    def react_to_true(self, bundle):
        '''Specialized reaction peforming the isolating.'''
        
        print('Isolating '+_format_b(bundle))
        
        try:
            bundle.graph.isolate(bundle.feature_space, layer=bundle.attribute)
        except:
            self.exception_recorder(bundle, context='Error while isolating in '+self.descr)
            
        try:
            if self.next_stage_true is not None:
                self.next_stage_true.socket.send(bundle)
        except:
            self.exception_recorder(bundle,
                                    context='Sending of bundle after isolation failed in '+self.descr)


class SplittingStage(LogicStage):
    '''Specialized stage to perform the splitting up of a clustered bundle of nodes
    into a new bundle for each cluster.'''
    
    def react_to_true(self, bundle):
        '''Specialized stage to split a bundle.
        Envokes IPAG.attribute_divided_fs_arrays().'''

        
        print('Splitting '+_format_b(bundle))
        
        try:
            new_fs_list = bundle.graph.attribute_divided_fs_arrays(bundle.attr_config,
                                                                   bundle.attribute,
                                                                   subset=bundle.feature_space.order)
            
#            for fs in new_fs_list:
#                print(fs.label, end=',  ')
        except:
            self.exception_recorder(bundle, context='Splitting of bundle failed in '+self.descr)
        
        try:
            #print(len(new_fs_list))

            if len(new_fs_list) == 1:
    
                new_bundle = proc_bundle(bundle.graph, bundle.attribute, bundle.attr_config,
                                         bundle.metric_config, new_fs_list[0], bundle.metric_dict)            
    
                self.react_to_false(new_bundle)
    
            else:
                for fs in new_fs_list:
                    metrics = bundle.graph.apply_group_metrics(fs, bundle.metric_config)
                    print(metrics)
    
                    new_bundle = proc_bundle(bundle.graph, bundle.attribute, bundle.attr_config,
                                             bundle.metric_config, fs, metrics)
    
                    if self.next_stage_true is not None:
                        if self.final_stage:
                            self.next_stage_true.queue_true.append((True, new_bundle, self.descr))
                        else:
                            self.next_stage_true.socket.send(bundle)

        except:
            self.exception_recorder(bundle,
                                    context='Processing of splitting output failed in '+self.descr)
            
                

class DynamicClustering(dict):
    '''Specialized dictionary to hold instances
    of LogicStage to facilitate their usage.'''
    
    def __init__(self, descr=None):

        if descr is not None:
            self.descr = descr
        else:
            self.descr = self.__repr__()
            
        self.exc_recorder = ExceptionRecorder('ExcRec: '+self.descr)
        self.set_exception_recorder(self.exc_recorder)
    
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

    def set_exception_recorder(self, exc_recorder=None):
        '''Set the exception recorder of all contained
        stages to a given instance.'''
        
        if exc_recorder is not None:
            assert(isinstance(exc_recorder, ExceptionRecorder))
            self.exc_recorder = exc_recorder
        
        for stage in self.values():
            stage.set_exception_recorder(self.exc_recorder)
            
    def bundle_queue(self):
        '''Providing the enqueued bundles sequentially
        prioritizing the ones coming from react_to_false().'''
        
        count_empty = 0       
        while count_empty < (len(self)*2):
            count_empty = 0
            dict_false = {}
            dict_true = {}
            for name, stage in self.items():
                try:
                    dict_false[name] = stage.queue_false.popleft()
                except IndexError:
                    count_empty += 1
                
                try:
                    dict_true[name] = stage.queue_true.popleft()
                except IndexError:
                    count_empty += 1
                
            yield from chain(dict_false.items(), dict_true.items())

        

    def __call__(self, graph, attr_config, attribute, metric_config, start_point):
        '''Conduct the dynamic clustering process.'''

        start_fs = graph.basic_feature_space_array(attr_config)
        
        metrics = graph.apply_group_metrics(start_fs, metric_config)
        
        start_bundle = proc_bundle(graph, attribute, attr_config, metric_config, start_fs, metrics)
        
        self[start_point].queue_true.append((True, start_bundle, '<START LOOP>'))
        
        for entry_point, element in self.bundle_queue():
            flag, bundle, sender = element
            m_data = (_format_b(bundle), entry_point, sender, flag)
            message = 'Handing over bundle {} to {} from {} - MODE:{}'.format(*m_data)
            print(message)
            self[entry_point].socket.send(element[1])

        





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
    
    er = ExceptionRecorder('Test er')
    
    er.raise_none_bundle_errors = False
    try:
        raise ValueError(42)
    except:
        er(mock_bundle)
    
    
    try:
        raise TimeoutError('bad luck')
    except:
        er()

