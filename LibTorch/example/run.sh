#!/bin/bash
set -e

pushd /workspace/LibTorch

mkdir -p build && rm -rf build/* # just make sure old build is removed
cmake -S example -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$OPENCV_INSTALL_DIR
cmake --build build --config Release
SPDLOG_LEVEL=info ./build/main

# popd