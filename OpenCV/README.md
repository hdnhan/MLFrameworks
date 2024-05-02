## Linux - Ubuntu
- Builing OpenCV from source: [here](https://docs.opencv.org/4.7.0/d2/de6/tutorial_py_setup_in_ubuntu.html)
- Check in [GPU Compute Capability](https://developer.nvidia.com/cuda-gpus)
```bash
python3 -c "import cv2; print(cv2.getBuildInformation())" # Check if OpenCV build information
python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())" # Check if CUDA is enabled

# CPU
docker build --progress=plain -t opencv-cpu -f dockerfiles/Dockerfile.cpu .
docker run --rm -it -v $(pwd)/..:/workspace opencv-cpu python3 /workspace/OpenCV/example/py/main.py
docker run --rm -it -v $(pwd)/..:/workspace opencv-cpu bash /workspace/OpenCV/example/run.sh

# CUDA
docker build --progress=plain -t opencv-cuda -f dockerfiles/Dockerfile.cuda .
docker run --rm -it --gpus all -v $(pwd)/..:/workspace opencv-cuda python3 /workspace/OpenCV/example/py/main.py
docker run --rm -it --gpus all -v $(pwd)/..:/workspace opencv-cuda bash /workspace/OpenCV/example/run.sh
```


## macOS - M2 Macbook Air
### Build OpenCV
```bash
bash build_opencv_osx.sh
```

### Python
```bash
python example/py/main.py
```

### C++
```bash
cmake -S example -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$OPENCV_INSTALL_DIR
cmake --build build --config Release
./build/main
```