# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 19:40:45 2017

@author: Henning
"""

#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
from libcpp cimport bool
import numpy as np
cimport numpy as cnp
#from libc.math cimport sin, cos, abs
#from interpolation cimport bilinear_interpolation, round
import cython

def _count_transitions(long[:, ::1] image):
    
    #old parameters
    cdef int n_sp = len(np.unique(image))
    cdef int chunk_size = n_sp
    cdef int start = 0

    output_shape = (chunk_size, n_sp)
    cdef int[:, ::1] output = np.zeros(output_shape, dtype=np.int32)

    cdef int[::1] roff = np.array([0, -1, 0, 1], dtype=np.int32)
    cdef int[::1] coff = np.array([-1, 0, 1, 0], dtype=np.int32)

    cdef int[::1] neighbours = np.array([-1, -1, -1, -1], dtype=np.int32)

    cdef Py_ssize_t rows = image.shape[0]
    cdef Py_ssize_t cols = image.shape[1]

    cdef Py_ssize_t r, c, n, r_index, c_index, r_out, c_cout

    with nogil:
        #STARTWERT IN RANGE 
        for r in range(image.shape[0]):
            for c in range(image.shape[1]):
    
                if image[r, c] < start:
                    continue
    
                for n in range(4):
                    r_index = r + roff[n]
                    c_index = c + coff[n]
    
                    if 0 <= r_index < rows and \
                    0 <= c_index < cols:
                        #print(r_index, c_index)
                        neighbours[n] = image[r_index, c_index]
    
                    else:
                        neighbours[n] = -1
                        
    
                for n in range(4):
                    if not neighbours[n] == -1:
                        c_out = <int>image[r, c]
                        r_out = <int>neighbours[n]
                        if not r_out == c_out:
                            output[r_out, c_out] += 1
                            #WIRD DOPPELT GEZÄHLT



    return np.asarray(output)
