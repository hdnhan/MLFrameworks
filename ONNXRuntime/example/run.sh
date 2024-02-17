#!/bin/bash

pushd /workspace/ONNXRuntime/example

cmake -S . -B /tmp/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$OPENCV_INSTALL_DIR -DORT_INSTALL_DIR=$ORT_INSTALL_DIR
cmake --build /tmp/build
/tmp/build/main

# popd