    #!/bin/bash
set -e
set -x

WORK_DIR=$PWD

# Prerequsite: conda, python 3.8 or above; cmake; LLVM-13

# Step 0: Create conda environment & install related packages
conda create -n tvm python=3.8 -y
conda activate tvm
pip install numpy decorator attrs tornado psutil xgboost cloudpickle pytest black pylint cpplint typing_extensions

# Step 1: Build TVM
git clone https://github.com/apache/tvm
cd tvm
git checkout 62f9b1d
git submodule update --init --recursive
mkdir build && cd build
cp ../cmake/config.cmake .
sed -i '/USE_LLVM/ s/OFF/\/llvm-project\/install\/bin\/llvm-config/g' config.cmake    # Enable LLVM
sed -i '/USE_OPENMP/ s/none/gnu/g' config.cmake    # Enable OpenMP
cmake .. && make -j$(nproc)
cd ..

# Step 2: Set TVM env
export TVM_HOME=`pwd`
export PYTHONPATH=${TVM_HOME}/python:${PYTHONPATH}
export TVM_LIBRARY_PATH=${TVM_HOME}/build
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /miniconda/envs/tvm/bin/../lib/libstdc++.so.6

# Step 3: back to TLCBench
cd ..