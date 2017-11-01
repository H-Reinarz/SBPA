# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 21:23:26 2017

@author: Jannik
"""

#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
from libcpp cimport bool
import numpy as np
cimport numpy as cnp
import cython

cdef extern from "numpy/npy_math.h":
    double NAN "NPY_NAN"

ctypedef fused any_int:
    cnp.uint8_t
    cnp.uint16_t
    cnp.uint32_t
    cnp.uint64_t
    cnp.int8_t
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t
    
def _graphLabels(int[:] values, theSet):
    #returnSet = set()
    cdef Py_ssize_t i = len(values) // 2
    cdef int center = values[i]
    
    for value in values:
        if value != center:
            theSet.add((center, value))
    
    #return returnSet
            
    
    