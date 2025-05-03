## Linux - Docker
```bash
# CPU
docker build --progress=plain -t libtorch-cpu -f dockerfiles/Dockerfile.cpu .
docker run --rm -it -v $(pwd)/..:/workspace libtorch-cpu python3 /workspace/LibTorch/example/py/main.py
docker run --rm -it -v $(pwd)/..:/workspace libtorch-cpu bash /workspace/LibTorch/example/run.sh
```

## macOS - M2 Macbook Air - Native
```bash
conda create -yn libtorch python=3.10 && conda activate libtorch
pip install torch~=2.7 opencv-python~=4.10.0

# Python
python example/py/main.py

# C++
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.7.0.zip
unzip libtorch-macos-arm64-2.7.0.zip  

export OPENCV_INSTALL_DIR=/tmp/opencv
cmake -S example -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$OPENCV_INSTALL_DIR
cmake --build build --config Release --parallel
SPDLOG_LEVEL=debug ./build/main
```
