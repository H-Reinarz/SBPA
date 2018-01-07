#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 16:16:39 2018

@author: hre070
"""

#IN DEVELOPMENT

class threshhold_stage(object):
    
    def __init__(self, threshhold_dict, **kwargs):
        
        self.threshhold_dict = threshhold_dict
        self.next_stage_true = None
        self.next_stage_false = None
        
      
    def set_successor_stages(self, true, false):
        self.next_stage_true = true
        self.next_stage_false = false

    def evaluate(self, metric_dict):
        evaluation = False
        
        for metric, thresh in self.threshhold_dict.items():
            if metric_dict[metric] > thresh:
                evaluation = True
                
        return evaluation
    
    def react_to_true(self, item):
        if self.next_stage_true is not None:
            self.next_stage_true.send(item)
    
    def react_to_false(self, item):
        if self.next_stage_false is not None:
            self.next_stage_false.send(item)
        
    def __call__(self):
        while True:
            item = yield
                    
            if self.evaluate(item):
                self.react_to_true(item)
            else:
                self.react_to_false(item)
                
class cluster_stage(threshhold_stage):
    pass