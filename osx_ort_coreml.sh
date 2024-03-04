#!/bin/bash

export ORT_INSTALL_DIR=$(pwd)/ort
export ORT_RELEASE_TAG=v1.16.3
mkdir -p $ORT_INSTALL_DIR

git clone --recursive -b $ORT_RELEASE_TAG https://github.com/microsoft/onnxruntime.git $ORT_INSTALL_DIR/onnxruntime

pip install cmake numpy packaging setuptools wheel
brew uninstall protobuf && brew cleanup # Uninstalling protobuf helps avoid some bugs
brew reinstall abseil # required 

cd $ORT_INSTALL_DIR/onnxruntime && \
./build.sh --config Release \
    --use_coreml \
    --compile_no_warning_as_error \
    --build_shared_lib \
    --build_wheel \
    --update \
    --parallel \
    --cmake_extra_defines CMAKE_INSTALL_PREFIX=$ORT_INSTALL_DIR

cd $ORT_INSTALL_DIR/onnxruntime/build/MacOS/Release && make install && \
python3 $ORT_INSTALL_DIR/onnxruntime/setup.py bdist_wheel

# For some reason, installed onnxruntime providers do not include coreml
cp $ORT_INSTALL_DIR/onnxruntime/include/onnxruntime/core/providers/coreml/* $ORT_INSTALL_DIR/include/onnxruntime

# # For some reason, macos_11_0 is not compatible with my system => rename it to macos_10_9
# cp dist/onnxruntime-1.16.3-cp38-cp38-macosx_11_0_x86_64.whl dist/onnxruntime-1.16.3-cp38-cp38-macosx_10_9_x86_64.whl
# pip3 install dist/onnxruntime-1.16.3-cp38-cp38-macosx_10_9_x86_64.whl
