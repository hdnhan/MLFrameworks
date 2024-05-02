#!/bin/bash

pushd /workspace/ONNXRuntime

cmake -S example -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$OPENCV_INSTALL_DIR -DORT_INSTALL_DIR=$ORT_INSTALL_DIR
cmake --build build --config Release
./build/main

# popd