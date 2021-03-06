#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 20:39:13 2017

@author: hre070
"""

from itertools import repeat
from math import ceil
import numpy as np



class Hist:
    '''Specialized histogramm class for use in
    region adjacency graphs as attribute representation.'''

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

        #Image and single value bins (i.e. LBP)
        elif len(args) == 1 and isinstance(args[0], np.ndarray) and 'vbins' in kwargs:
            self.keys = dict(zip(kwargs['vbins'], range(len(kwargs['vbins']))))

            if kwargs.__contains__("value"):
                value = kwargs["value"]
            else:
                value = 0

            self.container = np.array(list(repeat(value, len(kwargs['vbins']))))

            for pix in np.nditer([args[0]]):
                #individual bin incrementation
                self.increment(int(pix))

        #dict
        elif len(args) == 1 and isinstance(args[0], dict):
            input_dict = args[0]
            self.keys = dict(zip(input_dict.keys(), range(len(input_dict))))

            self.container = np.array(list(input_dict.values()))

        #image
        elif len(args) == 1 and isinstance(args[0], np.ndarray):
            values = args[0]
            #assert_nD(image, 2)

            self.keys = None

            if kwargs.__contains__("bins"):
                self.container, self.bins = np.histogram(values, kwargs["bins"])
            else:
                raise ValueError("Missing key word parameter 'bins'!")

        #Call with wrong arguments
        else:
            raise ValueError("Unusable parameter configuration. Please check source or docs!")

        #Normalize along with instance creation
        if kwargs.__contains__("normalize") and kwargs["normalize"] is True:
            if not kwargs.__contains__("pixel_count"):
                raise ValueError("Missing key word parameter 'pixel_count'!")
            else:
                self.normalize(kwargs["pixel_count"])
        else:
            self.is_normalized = False
            
            
    def normalize(self, denominator):
        """Normalize bin counts against a given number (i.e. pixel count)"""
        self.norm_container = np.zeros(len(self.container))
        for index, element in enumerate(np.nditer([self.container])):
            self.norm_container[index] = round(element/denominator*100, 3)

        #set falg attribute
        self.is_normalized = True


    def increment(self, key):
        """Increment a given bin by one"""
        self[key] += 1


    def __iadd__(self, other):
        """Incremental addition on counts"""
        for bin_, count in other:
            self[bin_] = self[bin_] + count
        return self


    def __iter__(self):
        """Iterate over bins and counts of the histogram"""
        if self.keys is not None:
            for key, val in self.keys.items():
                yield (key, self.container[val])
        else:
            yield from enumerate(self.container)


    def __getitem__(self, key):
        """Return count value for given bin"""
        if self.keys is not None:
            return self.container[self.keys[key]]
        else:
            return self.container[key]


    def __setitem__(self, key, value):
        """Set count value for given bin"""
        if self.keys is not None:
            self.container[self.keys[key]] = value
        else:
            self.container[key] = value


    def __call__(self, mode='array', normalized=True):
        """Return either a dictionary or numpy array representation
        of the class for use in functions.
        Parameter 'mode' must be either 'dict' or 'array'"""

        if normalized:
            cont = self.norm_container
        else:
            cont = self.container


        if self.keys is not None:
            keys = self.keys
        else:
            keys = range(len(self.container))


        if mode == "dict":
            return {k: cont[k] for k in keys}
        elif mode == "array":
            return cont
        else:
            raise ValueError("Parameter 'mode' must be either 'dict' or 'array'!")


    def __str__(self):
        """Return formatted string of the histogram"""

        fill = 8
        per_ln = 12
        starts = ("K: ", "C: ", "N: ")

        if self.keys is not None:
            keys = [str(k).ljust(fill) for k in self.keys]
        else:
            keys = [str(k).ljust(fill) for k in range(len(self.container))]

        counts = [str(e).ljust(fill) for e in self.container]

        if self.is_normalized:
            norms = [str(e).ljust(fill) for e in self.norm_container]
            elements = list(zip(keys, counts, norms))
        else:
            elements = list(zip(keys, counts))

        lines = []

        for _ in range(ceil(len(elements)/per_ln)):
            lines.append(elements[:per_ln])
            elements = elements[per_ln:]

        str_lines = []
        for line in lines:
            sub_lines = []
            for index in range(len(line[0])):
                sub_ln = starts[index] + " ".join([str(e[index]) for e in line])
                sub_lines.append(sub_ln)
            str_lines.append("\n".join(sub_lines)+ "\n" + "="*(fill*per_ln+10))

        return "\n".join(str_lines)
