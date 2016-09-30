__author__ = 'arenduchintala'
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy 
setup(
        name='c array utils',
        ext_modules=cythonize('c_array_utils.pyx'),
        include_dirs=[numpy.get_include()]
        )

#setup(
#        name='c LBP',
#        ext_modules=cythonize('c_LBP.pyx'),
#        include_dirs=[numpy.get_include()]
#        )


# Build:
# python setup.py build_ext --inplace

