#==============================================================================
# MODIFIED CODE BASED ON: skimagge.feature.setup.py
#==============================================================================
import os
from skimage._build import cython

base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('LBP_BINS', parent_package, top_path)
    #config.add_data_dir('tests')

    cython(['lbp_bins.pyx'], working_path=base_path)

    config.add_extension('lbp_bins', sources=['lbp_bins.c'],
                         include_dirs=[get_numpy_include_dirs()])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer='Henning Reinarz',
          description='LBP_BINS',
          **(configuration(top_path='').todict())
          )
