# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 20:17:13 2017

@author: Jannik
"""

import os
from skimage._build import cython

base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('LBP', parent_package, top_path)
    #config.add_data_dir('tests')

    cython(['lbp.pyx'], working_path=base_path)

    config.add_extension('lbp', sources=['lbp.c'],
                         include_dirs=[get_numpy_include_dirs()])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer='Jannik Guenther',
          description='LBP',
          **(configuration(top_path='').todict())
          )