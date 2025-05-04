#!/bin/bash

export MNN_INSTALL_DIR=/tmp/mnn
export MNN_RELEASE_TAG=3.1.3

# Create a (miniconda) virtual environment and activate it
pip install "cmake<4" numpy
# https://github.com/alibaba/MNN/blob/master/pymnn/pip_package/setup.py
pip install MNN~=$MNN_RELEASE_TAG

git clone --recursive -b $MNN_RELEASE_TAG https://github.com/alibaba/MNN.git /tmp/git/MNN

cd /tmp/git/MNN &&  ./schema/generate.sh && \
mkdir -p build && cd build && \
cmake -S .. -B . \
    -DMNN_DEBUG=OFF \
    -DMNN_METAL=ON \
    -DMNN_COREML=ON \
    -DCMAKE_INSTALL_PREFIX=$MNN_INSTALL_DIR && \
make -j6 && make install
