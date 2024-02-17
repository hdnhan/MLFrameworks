#!/bin/bash

pushd /workspace/OpenCV/example

cmake -S . -B /tmp/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$OPENCV_INSTALL_DIR
cmake --build /tmp/build
/tmp/build/main

# popd