#!/bin/bash

# Load python 2.7
module load python/2.7

# Activate virtual environment
virtualenv-2.7 python_virtenv_demo
source python_virtenv_demo/bin/activate

# Install python packages (some are unused in this demo)
pip install numpy
pip install scipy
pip install scikit-learn
pip install matplotlib
pip install h5py
pip install pandas

# Deactivate virtual environment
deactivate

# Compress virtual environment
tar -cvzf python_virtenv_demo.tar.gz python_virtenv_demo

