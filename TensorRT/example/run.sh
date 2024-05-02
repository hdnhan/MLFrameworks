#!/bin/bash

pushd /workspace/TensorRT

cmake -S example -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$OPENCV_INSTALL_DIR
cmake --build build
./build/main

# popd