#!/bin/bash

export ORT_INSTALL_DIR=/tmp/onnxruntime
export ORT_RELEASE_TAG=v1.21.1
mkdir -p $ORT_INSTALL_DIR

git clone --recursive -b $ORT_RELEASE_TAG https://github.com/microsoft/onnxruntime.git /tmp/git/onnxruntime

pip install "cmake<4" numpy packaging setuptools wheel
# brew uninstall protobuf && brew cleanup # Uninstalling protobuf helps avoid some bugs
# brew reinstall abseil # required 

cd /tmp/git/onnxruntime && \
./build.sh --config Release \
    --use_coreml \
    --compile_no_warning_as_error \
    --build_shared_lib \
    --build_wheel \
    --update \
    --parallel \
    --skip_tests \
    --cmake_extra_defines \
    CMAKE_INSTALL_PREFIX=$ORT_INSTALL_DIR \
    onnxruntime_BUILD_UNIT_TESTS=OFF \
    PYTHON_EXECUTABLE=$(which python) Python_EXECUTABLE=$(which python)

cd /tmp/git/onnxruntime/build/MacOS/Release && make -j6 && make install && \
python /tmp/git/onnxruntime/setup.py bdist_wheel && \
pip install /tmp/git/onnxruntime/build/MacOS/Release/dist/*.whl
pip install opencv-python~=4.10.0
