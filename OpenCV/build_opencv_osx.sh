#!/bin/bash

export OPENCV_INSTALL_DIR=$(pwd)/opencv
export OPENCV_RELEASE_TAG=4.9.0
mkdir -p $OPENCV_INSTALL_DIR

git clone --recursive -b $OPENCV_RELEASE_TAG https://github.com/opencv/opencv.git $OPENCV_INSTALL_DIR/git/opencv
git clone --recursive -b $OPENCV_RELEASE_TAG https://github.com/opencv/opencv_contrib.git $OPENCV_INSTALL_DIR/git/opencv_contrib

# Create a (miniconda) virtual environment and activate it
pip install cmake numpy
brew install pkg-config ffmpeg # check: brew list

cd $OPENCV_INSTALL_DIR/git/opencv && mkdir -p build && cd build && \
cmake -S .. -B . \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX=$OPENCV_INSTALL_DIR \
    -DINSTALL_PYTHON_EXAMPLES=OFF \
    -DINSTALL_C_EXAMPLES=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DOPENCV_ENABLE_NONFREE=ON \
    -DPYTHON3_EXECUTABLE=$(which python) \
    -DOPENCV_EXTRA_MODULES_PATH=$OPENCV_INSTALL_DIR/git/opencv_contrib/modules && \
make -j8 && make install

# Create a symbolic link 
ln -s $OPENCV_INSTALL_DIR/lib/python3.10/site-packages/cv2 $(python -c 'import site; print(site.getsitepackages()[0])')/cv2
