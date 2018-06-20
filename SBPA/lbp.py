# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 19:40:47 2017

@author: Jannik
"""

"""
Methods to characterize image textures.
"""

import numpy as np
from skimage._shared.utils import assert_nD


from ._LBP import _lbp, _lbp_rad, _lbp_ang, _nilbp


def local_binary_pattern(image, textureMap, P, R, method='default', nilbp = False):
    """Gray scale and rotation invariant LBP (Local Binary Patterns).

    LBP is an invariant descriptor that can be used for texture classification.

    Parameters
    ----------
    image : (N, M) array
        Graylevel image.
    P : int
        Number of circularly symmetric neighbour set points (quantization of
        the angular space).
    R : float
        Radius of circle (spatial resolution of the operator).
    method : {'default', 'ror', 'uniform', 'var'}
        Method to determine the pattern.

        * 'default': original local binary pattern which is gray scale but not
            rotation invariant.
        * 'ror': extension of default implementation which is gray scale and
            rotation invariant.
        * 'uniform': improved rotation invariance with uniform patterns and
            finer quantization of the angular space which is gray scale and
            rotation invariant.
        * 'nri_uniform': non rotation-invariant uniform patterns variant
            which is only gray scale invariant [2]_.
        * 'var': rotation invariant variance measures of the contrast of local
            image texture which is rotation but not gray scale invariant.

    Returns
    -------
    output : (N, M) array
        LBP image.

    References
    ----------
    .. [1] Multiresolution Gray-Scale and Rotation Invariant Texture
           Classification with Local Binary Patterns.
           Timo Ojala, Matti Pietikainen, Topi Maenpaa.
           http://www.rafbis.it/biplab15/images/stories/docenti/Danielriccio/Articoliriferimento/LBP.pdf, 2002.
    .. [2] Face recognition with local binary patterns.
           Timo Ahonen, Abdenour Hadid, Matti Pietikainen,
           http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.214.6851,
           2004.
    """
    
    if textureMap is None:
        textureMap = image
    
    assert_nD(image, 2)
    assert_nD(textureMap, 2)

    methods = {
        'default': ord('D'),
        'ror': ord('R'),
        'uniform': ord('U'),
        'nri_uniform': ord('N'),
        'var': ord('V'),
    }
    
    if nilbp:
        nilbp = 1
    else:
        nilbp = 0
        
    image = np.ascontiguousarray(image, dtype=np.double)
    textureMap = np.ascontiguousarray(image, dtype=np.double)
    output = _lbp._local_binary_pattern(image, textureMap, P, R, methods[method.lower()], nilbp)
    return output



def ni_lbp(image, P, R, method='default'):
    """Neighbour Intensity LBP"""
    
    assert_nD(image, 2)

    methods = {
        'default': ord('D'),
        'ror': ord('R'),
        'uniform': ord('U'),
        'nri_uniform': ord('N'),
        'var': ord('V'),
    }
    
        
    image = np.ascontiguousarray(image, dtype=np.double)
    output = _nilbp._ni_lbp(image, P, R, methods[method.lower()])
    return output



def radial_lbp(image, P, R_OUT, R_IN, method='default'):
    """Radial Difference LBP"""
    
    assert_nD(image, 2)

    methods = {
        'default': ord('D'),
        'ror': ord('R'),
        'uniform': ord('U'),
        'nri_uniform': ord('N'),
        'var': ord('V'),
    }
            
    image = np.ascontiguousarray(image, dtype=np.double)
    output = _lbp_rad._radial_lbp(image, P, R_OUT, R_IN, methods[method.lower()])
    return output



def angular_lbp(image, P, R, method='default'):
    """Angular Difference LBP"""
    
    assert_nD(image, 2)

    methods = {
        'default': ord('D'),
        'ror': ord('R'),
        'uniform': ord('U'),
        'nri_uniform': ord('N'),
        'var': ord('V'),
    }
            
    image = np.ascontiguousarray(image, dtype=np.double)
    output = _lbp_ang._angular_lbp(image, P, R, methods[method.lower()])
    return output