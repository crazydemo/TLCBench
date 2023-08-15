config TVM python env
git clone --recursive https://github.com/apache/tvm && cd tvm    # clone tvm

conda env create --name tvm --file conda/build-environment.yaml    # install pre-requisites

conda activate tvm

pip install numpy decorator attrs tornado psutil xgboost cloudpickle pytest black pylint cpplint typing_extensions

pip install mxnet==1.8.0.post0 gluoncv    # optional

pip install ffi-navigator ipykernel    # optional

build TVM
mkdir build && cd build

cp ../cmake/config.cmake .

sed -i '/USE_LLVM/ s/OFF/ON/g' config.cmake    # Enable LLVM

sed -i '/USE_OPENMP/ s/none/gnu/g' config.cmake    # Enable OpenMP

cmake .. && make -j$(nproc)

cd ..

About libdnnl.so

If cmake report that EXTERN_LIBRARY_DNNL NOTFOUND, we need to set it by add a parameter "-DEXTERN_LIBRARY_DNNL=<conda root>/envs/tvm/lib/libdnnl.so" to cmake.

setup TVM env
export TVM_HOME=`pwd`

export PYTHONPATH=${TVM_HOME}/python:${PYTHONPATH}

export TVM_LIBRARY_PATH=${TVM_HOME}/build
