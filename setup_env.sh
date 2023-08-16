#!/bin/bash
set -e
set -x

WORK_DIR=$PWD
CONDA_ENV_NAME=tvm

# Prerequsite: conda, python 3.8 or above; cmake; LLVM-13

# Step 1: Create conda environment & install related packages
conda create -n ${CONDA_ENV_NAME} python=3.8 -y
conda activate ${CONDA_ENV_NAME}

pip install numpy decorator attrs tornado psutil xgboost cloudpickle pytest black pylint cpplint typing_extensions

# Step 2: Build tvm
git clone --recursive https://github.com/apache/tvm
cd tvm
mkdir build && cd build
cp ../cmake/config.cmake .
sed -i '/USE_LLVM/ s/OFF/ON/g' config.cmake    # Enable LLVM
sed -i '/USE_OPENMP/ s/none/gnu/g' config.cmake    # Enable OpenMP
cmake .. && make -j$(nproc)
cd ..

# Step 3: setup tvm env
export TVM_HOME=`pwd`
export PYTHONPATH=${TVM_HOME}/python:${PYTHONPATH}
export TVM_LIBRARY_PATH=${TVM_HOME}/build