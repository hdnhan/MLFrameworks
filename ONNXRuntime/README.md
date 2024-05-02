Note: This requires to build OpenCV first (except TensorRT and OpenVINO).

## Linux - Docker
```bash
cd path/CVExamples/ONNXRuntime

# CPU
docker build --progress=plain -t onnxruntime-cpu -f dockerfiles/Dockerfile.cpu .
docker run --rm -it -v $(pwd)/..:/workspace onnxruntime-cpu python3 /workspace/ONNXRuntime/example/py/main.py
docker run --rm -it -v $(pwd)/..:/workspace onnxruntime-cpu bash /workspace/ONNXRuntime/example/run.sh

# CUDA
docker build --progress=plain -t onnxruntime-cuda -f dockerfiles/Dockerfile.cuda .
docker run --rm -it --gpus all -v $(pwd)/..:/workspace onnxruntime-cuda python3 /workspace/ONNXRuntime/example/py/main.py
docker run --rm -it --gpus all -v $(pwd)/..:/workspace onnxruntime-cuda bash /workspace/ONNXRuntime/example/run.sh

# TensorRT
docker build --progress=plain -t onnxruntime-tensorrt -f dockerfiles/Dockerfile.tensorrt .
docker run --rm -it --gpus all -v $(pwd)/..:/workspace onnxruntime-tensorrt python3 /workspace/ONNXRuntime/example/py/main.py
docker run --rm -it --gpus all -v $(pwd)/..:/workspace onnxruntime-tensorrt bash /workspace/ONNXRuntime/example/run.sh

# OpenVINO
docker build --progress=plain -t onnxruntime-openvino -f dockerfiles/Dockerfile.openvino .
docker run --rm -it -v $(pwd)/..:/workspace onnxruntime-openvino python3 /workspace/ONNXRuntime/example/py/main.py
docker run --rm -it -v $(pwd)/..:/workspace onnxruntime-openvino bash /workspace/ONNXRuntime/example/run.sh
```

## macOS - M2 Macbook Air
```bash
cd path/CVExamples/ONNXRuntime

# Build OpenCV
bash build_onnxruntime_osx.sh

# Python
python example/py/main.py

# C++
cmake -S example -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$OPENCV_INSTALL_DIR -DORT_INSTALL_DIR=$ORT_INSTALL_DIR
cmake --build build --config Release
./build/main
```