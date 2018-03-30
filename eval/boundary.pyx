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




def _get_length(int [:, ::1] region):
    
    cdef long boundary = 0

    cdef int[::1] roff = np.array([0, -1, 0, 1], dtype=np.int32)
    cdef int[::1] coff = np.array([-1, 0, 1, 0], dtype=np.int32)

    cdef Py_ssize_t rows = region.shape[0]
    cdef Py_ssize_t cols = region.shape[1]

    cdef Py_ssize_t r, c, n, r_index, c_index, r_out, c_cout

    with nogil:
        #STARTWERT IN RANGE 
        for r in range(region.shape[0]):
            for c in range(region.shape[1]):
    
                for n in range(4):
                    r_index = r + roff[n]
                    c_index = c + coff[n]
    
                    if region[r, c] == 1:
                        #print(r_index, c_index)
                        if not 0 <= r_index < rows or not 0 <= c_index < cols \
                        or region[r_index, c_index] == 0:
                                boundary += 1
    return boundary
