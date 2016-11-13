from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(['adaboost_funcs.pyx', 'haar_funcs.pyx', 'realboost_funcs.pyx'])
)