# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 19:40:45 2017

@author: Jannik
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



def _count_transitions(int[:, ::1] image, int n_sp, int chunk_size, int start):


    output_shape = (chunk_size, n_sp)
    cdef int[:, ::1] output = np.zeros(output_shape, dtype=np.int32)

    cdef int[::1] roff = np.array([0, -1, 0, 1])
    cdef int[::1] coff = np.array([-1, 0, 1, 0])

    cdef int[::1] neighbours = np.array([-1, -1, -1, -1], dtype=np.int64)

    cdef Py_ssize_t rows = image.shape[0]
    cdef Py_ssize_t cols = image.shape[1]

    cdef Py_ssize_t r, c, n, r_index, c_index, r_out, c_cout

    with nogil:
        for r in range(image.shape[0]):
            for c in range(image.shape[1]):

                if image[r, c] < start:
                    continue

                for n in range(4):
                    r_index = r + roff[n]
                    c_index = c + coff[n]

                    if not 0 < r_index < rows and \
                    not 0 < c_index < cols:
                        neighbours[n] = -1

                    else:
                        neighbours[n] = image[r_index, c_index]

                for n in range(4):
                    c_out = <int>image[r, c]
                    r_out = <int>neighbours[n]
                    output[r_out, c_out] += 1



    return np.asarray(output)
