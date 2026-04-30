import sys, os
from setuptools import setup, find_packages, Extension
import numpy

#ensure Cython is installed before importing it
try:
    from Cython.Build import cythonize
except ImportError:
    print("Cython not found, installing it now...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "Cython"], check=True, capture_output=True)
    from Cython.Build import cythonize

setup(
    name             = 'pyEFPEHM',
    description      = 'Python package for an efficient fully precessing eccentric inspiral model.',
    version          = '0.1',
    author           = 'Gonzalo Morrás, Geraint Pratten, Patricia Schmidt',
    author_email     = 'gonzalo.morras@ligo.org',
    packages         = find_packages(),
    package_dir      = {'pyEFPEHM': 'pyEFPEHM'},
    ext_modules      = cythonize([Extension("pyEFPEHM.utils.cython_utils",["pyEFPEHM/utils/cython_utils.pyx"],include_dirs=[numpy.get_include()]),]),
    url              = '',
    download_url     = '',
    install_requires = ['numpy','scipy>=1.9.3','Cython'],
)

