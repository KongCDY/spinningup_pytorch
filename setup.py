from os.path import join, dirname, realpath
from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "The Spinning Up repo is designed to work with Python 3.6 and greater." \
    + "Please install it before proceeding."

with open(join("spinup_pt", "version.py")) as version_file:
    exec(version_file.read())

setup(
    name='spinup_pt',
    py_modules=['spinup_pt'],
    version=__version__,#'0.1',
    install_requires=[
        'cloudpickle==0.5.2',
        'gym[atari,box2d,classic_control]>=0.10.8',
        'ipython',
        'joblib',
        'matplotlib==3.0.2',
        'mpi4py',
        'numpy',
        'pandas',
        'pytest',
        'psutil',
        'scipy',
        'seaborn==0.8.1',
        'torch',
        'tqdm'
    ],
    extras_require={'mujoco': 'mujoco-py<2.1,>=2.0'},
    description="A Pytorch version of Spinning Up, which is a teaching tools for introducing people to deep RL.",
    author="XingChen",
)
