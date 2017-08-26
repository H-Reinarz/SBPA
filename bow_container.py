#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 20:39:13 2017

@author: hre070
"""

import numpy as np
from skimage._shared.utils import assert_nD
from itertools import repeat
from math import ceil



class hist:
    def __init__(self, *args, **kwargs):

        #Bins
        if len(args) == 1 and isinstance(args[0], set):
            bins = args[0]
            self.keys = dict(zip(bins, range(len(bins))))
            
            if kwargs.__contains__("value"):
                value = kwargs["value"]
            else:
                value = 0
                        
            self.container = np.array(list(repeat(value, len(bins))))
            
            
        #dict
        elif len(args) == 1 and isinstance(args[0], dict):
            input_dict = args[0]
            self.keys = dict(zip(input_dict.keys(), range(len(input_dict))))
            
            self.container = np.array(list(input_dict.values()))
            

        #image
        elif len(args) == 1 and isinstance(args[0], np.ndarray):
            image = args[0]
            assert_nD(image, 2)
            
            self.keys = None
            
            self.container, self.bins  = np.histogram(image, kwargs["bins"])        
            
        #Call with wrong arguments
        else:
            raise ValueError


        #Normalize along with instance creation
        if kwargs["normalize"]:
            if not kwargs.__contains__("pixel_count"):
                raise ValueError
            else:
                self.normalize(kwargs["pixel count"])
        else:
           self.is_normalized = False


    
    def normalize(self, n):
        self.norm_container = np.zeros(len(self.container))
        for ix, e in enumerate(np.nditer([self.container])):
            self.norm_container[ix] = round((e/n*100, 3))
            
        #set falg attribute
        self.is_normalized = True

    
    def increment(self, key):
        self[key] += 1

    
    def __iter__(self):
        if self.keys is not None:
            for k, v in self.keys.items():
                yield (k, self.container[v])                        
            pass
        else:
            yield from enumerate(self.container)

    
    def __getitem__(self, key):
        if self.keys is not None:
            return self.container[self.keys[key]]
        else:
            return self.container[key]
        pass

    def __setitem__(self, key, value):
        if self.keys is not None:
            self.container[self.keys[key]] = value
        else:
            self.container[key] = value



    def __call__(self, mode):
        if mode == "dict":
            return {k: self.container[k] for k in self.keys}
        elif mode == "array":
            return self.container
        else:
            raise ValueError

    
    def __str__(self):
        k = "K: "
        c = "C: "
        n = "N: "
        
        k_line1 = k + " ".join([str(k).ljust(6) for k in self.keys()])
        c_line1 = c + " ".join([str(e).ljust(6) for e in self.container])
        
        if self.is_normalized:
            n_line1 = n + " ".join([str(e).ljust(6) for e in self.norm_container])
            
        remaining = len(k_line1)
            
        while remaining > 100:
            pass
            
        
        
        pass
    

#test = hist()