#==============================================================================
# MODIFIED CODE BASED ON: skimagge.feature.setup.py
#==============================================================================
import os
from skimage._build import cython

base_path = os.path.abspath(os.path.dirname(__file__))


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

    config = Configuration('_LBP', parent_package, top_path)
    #config.add_data_dir('tests')

    cython(['src/lbp_bins.pyx'], working_path=base_path)
    cython(['src/_lbp.pyx'], working_path=base_path)
    cython(['src/_lbp_ang.pyx'], working_path=base_path)
    cython(['src/_lbp_rad.pyx'], working_path=base_path)
    cython(['src/_nilbp.pyx'], working_path=base_path)
    cython(['src/_sp_neighbors.pyx'], working_path=base_path)
    cython(['src/boundary.pyx'], working_path=base_path)

    config.add_extension('lbp_bins', sources=['src/lbp_bins.c'],
                         include_dirs=[get_numpy_include_dirs()])
						 #include_dirs=[get_numpy_include_dirs()])
    config.add_extension('_lbp', sources=['src/_lbp.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('_lbp_ang', sources=['src/_lbp_ang.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('_lbp_rad', sources=['src/_lbp_rad.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('_nilbp', sources=['src/_nilbp.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('_sp_neighbors', sources=['src/_sp_neighbors.c'],
                         include_dirs=[get_numpy_include_dirs()])
    config.add_extension('boundary', sources=['src/boundary.c'],
                         include_dirs=[get_numpy_include_dirs()])

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(maintainer='Henning Reinarz and Jannik Guenther',
          description='LBP_BINS AND LBP',
          **(configuration(top_path='').todict())
          )
