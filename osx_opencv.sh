#!/bin/bash

export OPENCV_INSTALL_DIR=$(pwd)/ocv
export OPENCV_RELEASE_TAG=4.7.0
mkdir -p $OPENCV_INSTALL_DIR

git clone --recursive -b $OPENCV_RELEASE_TAG https://github.com/opencv/opencv.git $OPENCV_INSTALL_DIR/opencv
git clone --recursive -b $OPENCV_RELEASE_TAG https://github.com/opencv/opencv_contrib.git $OPENCV_INSTALL_DIR/opencv_contrib

pip install cmake numpy
brew install pkg-config ffmpeg

cd $OPENCV_INSTALL_DIR/opencv && mkdir -p build && cd build && \
cmake -S .. -B . \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX=$OPENCV_INSTALL_DIR \
    -DINSTALL_PYTHON_EXAMPLES=OFF \
    -DINSTALL_C_EXAMPLES=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DOPENCV_ENABLE_NONFREE=ON \
    -DOPENCV_EXTRA_MODULES_PATH=$OPENCV_INSTALL_DIR/opencv_contrib/modules && \
make -j8 && make install

# For some reason, there is not whl file for opencv-python
pip install opencv-python-headless
