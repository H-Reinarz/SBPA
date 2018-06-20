#==============================================================================
# MODIFIED CODE BASED ON: skimagge.feature._texture.pyx
#==============================================================================



#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as cnp
import cython


cdef inline int _bit_rotate_right(int value, int length) nogil:
    """Cyclic bit shift to the right.

    Parameters
    ----------
    value : int
        integer value to shift
    length : int
        number of bits of integer

    """
    return (value >> 1) | ((value & 1) << (length - 1))



def signed_tex_gen(n):
    for i in range(2**n):
        as_text = '{:0{}b}'.format(i, n)
        yield np.array([int(c) for c in as_text], dtype=np.int8)


def lbp_bins(int P, method='default'):
    """Function to compute all possible values produced by a call of 
    skimage.feature.texture.local_binary_pattern() with the same parameters.
    The output set provides the bins for BOW histograms.

    Parameters
    ----------
    P : int
        Number of circularly symmetric neighbour set points (quantization of

    method : {'D', 'R', 'U', 'N', 'V'}
        Method to determine the pattern.

        * 'D': 'default'
        * 'R': 'ror'
        * 'U': 'uniform'
        * 'N': 'nri_uniform'

    Returns
    -------
    output : Set of bins to constract BOW RAG
    """



    # texture weights
    cdef int[::1] weights = 2 ** np.arange(P, dtype=np.int32)

    # pre-allocate arrays for computation
    cdef signed char[::1] signed_texture = np.zeros(P, dtype=np.int8)
    cdef int[::1] rotation_chain = np.zeros(P, dtype=np.int32)

    cdef double lbp
    cdef Py_ssize_t r, c, changes, i
    cdef Py_ssize_t rot_index, n_ones
    cdef cnp.int8_t first_zero, first_one

    
    #Return object
    value_set = set()

    #set _method  
    methods = {
        'default': ord('D'),
        'ror': ord('R'),
        'uniform': ord('U'),
        'nri_uniform': ord('N'),
        'var': ord('V')
    }

    cdef char _method = methods[method.lower()]


    #with nogil:
    for tex in signed_tex_gen(P):
        signed_texture = tex
        
        
        lbp = 0

        if _method == 'U' or _method == 'N':
            # determine number of 0 - 1 changes
            changes = 0
            for i in range(P - 1):
                changes += (signed_texture[i]
                            - signed_texture[i + 1]) != 0
            if _method == 'N':
                # Uniform local binary patterns are defined as patterns
                # with at most 2 value changes (from 0 to 1 or from 1 to
                # 0). Uniform patterns can be characterized by their
                # number `n_ones` of 1.  The possible values for
                # `n_ones` range from 0 to P.
                #
                # Here is an example for P = 4:
                # n_ones=0: 0000
                # n_ones=1: 0001, 1000, 0100, 0010
                # n_ones=2: 0011, 1001, 1100, 0110
                # n_ones=3: 0111, 1011, 1101, 1110
                # n_ones=4: 1111
                #
                # For a pattern of size P there are 2 constant patterns
                # corresponding to n_ones=0 and n_ones=P. For each other
                # value of `n_ones` , i.e n_ones=[1..P-1], there are P
                # possible patterns which are related to each other
                # through circular permutations. The total number of
                # uniform patterns is thus (2 + P * (P - 1)).

                # Given any pattern (uniform or not) we must be able to
                # associate a unique code:
                #
                # 1. Constant patterns patterns (with n_ones=0 and
                # n_ones=P) and non uniform patterns are given fixed
                # code values.
                #
                # 2. Other uniform patterns are indexed considering the
                # value of n_ones, and an index called 'rot_index'
                # reprenting the number of circular right shifts
                # required to obtain the pattern starting from a
                # reference position (corresponding to all zeros stacked
                # on the right). This number of rotations (or circular
                # right shifts) 'rot_index' is efficiently computed by
                # considering the positions of the first 1 and the first
                # 0 found in the pattern.

                if changes <= 2:
                    # We have a uniform pattern
                    n_ones = 0  # determines the number of ones
                    first_one = -1  # position was the first one
                    first_zero = -1  # position of the first zero
                    for i in range(P):
                        if signed_texture[i]:
                            n_ones += 1
                            if first_one == -1:
                                first_one = i
                        else:
                            if first_zero == -1:
                                first_zero = i
                    if n_ones == 0:
                        lbp = 0
                    elif n_ones == P:
                        lbp = P * (P - 1) + 1
                    else:
                        if first_one == 0:
                            rot_index = n_ones - first_zero
                        else:
                            rot_index = P - first_one
                        lbp = 1 + (n_ones - 1) * P + rot_index
                else:  # changes > 2
                    lbp = P * (P - 1) + 2
            else:  # _method != 'N'
                if changes <= 2:
                    for i in range(P):
                        lbp += signed_texture[i]
                else:
                    lbp = P + 1
        else:
            # _method == 'default'
            for i in range(P):
                lbp += signed_texture[i] * weights[i]

            # _method == 'ror'
            if _method == 'R':
                # shift LBP P times to the right and get minimum value
                rotation_chain[0] = <int>lbp
                for i in range(1, P):
                    rotation_chain[i] = \
                        _bit_rotate_right(rotation_chain[i - 1], P)
                lbp = rotation_chain[0]
                for i in range(1, P):
                    lbp = min(lbp, rotation_chain[i])

        value_set.add(lbp)

    return value_set


