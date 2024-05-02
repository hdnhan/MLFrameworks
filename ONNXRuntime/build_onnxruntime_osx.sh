#!/bin/bash

export ORT_INSTALL_DIR=$(pwd)/onnxruntime
export ORT_RELEASE_TAG=v1.16.3
mkdir -p $ORT_INSTALL_DIR

git clone --recursive -b $ORT_RELEASE_TAG https://github.com/microsoft/onnxruntime.git $ORT_INSTALL_DIR/git/onnxruntime

pip install cmake numpy packaging setuptools wheel
# brew uninstall protobuf && brew cleanup # Uninstalling protobuf helps avoid some bugs
# brew reinstall abseil # required 

cd $ORT_INSTALL_DIR/git/onnxruntime && \
./build.sh --config Release \
    --use_coreml \
    --compile_no_warning_as_error \
    --build_shared_lib \
    --build_wheel \
    --update \
    --parallel \
    --cmake_extra_defines CMAKE_INSTALL_PREFIX=$ORT_INSTALL_DIR \
    PYTHON_EXECUTABLE=$(which python) Python_EXECUTABLE=$(which python)

cd $ORT_INSTALL_DIR/git/onnxruntime/build/MacOS/Release && make install && \
python $ORT_INSTALL_DIR/git/onnxruntime/setup.py bdist_wheel && \
pip install $ORT_INSTALL_DIR/git/onnxruntime/build/MacOS/Release/dist/*.whl

# For some reason, installed onnxruntime providers do not include coreml
cp $ORT_INSTALL_DIR/git/onnxruntime/include/onnxruntime/core/providers/coreml/* $ORT_INSTALL_DIR/include/onnxruntime
