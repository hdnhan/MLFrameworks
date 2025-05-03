#!/bin/bash

export OPENCV_INSTALL_DIR=/tmp/opencv
export OPENCV_RELEASE_TAG=4.10.0

git clone --recursive -b $OPENCV_RELEASE_TAG https://github.com/opencv/opencv.git /tmp/git/opencv
git clone --recursive -b $OPENCV_RELEASE_TAG https://github.com/opencv/opencv_contrib.git /tmp/git/opencv_contrib

# Create a (miniconda) virtual environment and activate it
pip install "cmake<4" numpy
brew install pkg-config ffmpeg # check: brew list

cd /tmp/git/opencv && mkdir -p build && cd build && \
cmake -S .. -B . \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX=$OPENCV_INSTALL_DIR \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DINSTALL_PYTHON_EXAMPLES=OFF \
    -DINSTALL_C_EXAMPLES=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DOPENCV_ENABLE_NONFREE=ON \
    -DPYTHON3_EXECUTABLE=$(which python) \
    -DOPENCV_EXTRA_MODULES_PATH=/tmp/git/opencv_contrib/modules && \
make -j6 && make install

# Create a symbolic link (not sure why not working)
# ln -sf $OPENCV_INSTALL_DIR/lib/python3.10/site-packages/cv2 $(python -c 'import site; print(site.getsitepackages()[0])')/cv2
pip install opencv-python~=$OPENCV_RELEASE_TAG
