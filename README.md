# Computer Vision Examples
```python
# Generate ONNX model from PyTorch
cd Assets && python export.py
```

Run inference on `yolov8n.onnx` and `g4dn.xlarge` (NVIDIA T4) using
- OpenCV: CPU and CUDA
- ONNXRuntime with different Execution Providers: CPU, CUDA, CoreML, OpenVINO, TensorRT
- TensorRT

## Download video
```bash
wget https://media.roboflow.com/supervision/video-examples/people-walking.mp4 -O Assets/video.mp4
mkdir -p Results
```

## Memory Check
[Valgrind docs](https://web.stanford.edu/class/archive/cs/cs107/cs107.1222/resources/valgrind.html)
```bash
apt install -y valgrind
cmake -S . -B /tmp/build -DCMAKE_BUILD_TYPE=Debug ...
cmake --build /tmp/build
valgrind --leak-check=full \
         --show-leak-kinds=all \
         --track-origins=yes \
         --suppressions=$OPENCV_INSTALL_DIR/share/opencv4/valgrind.supp \
         --suppressions=$OPENCV_INSTALL_DIR/share/opencv4/valgrind_3rdparty.supp \
         /tmp/build/main
# cuda-memcheck --leak-check full --show-backtrace yes --log-file /tmp/gpu.out [executable/python3 main.py]
compute-sanitizer --tool memcheck --leak-check full --show-backtrace yes --log-file /tmp/gpu.out [executable/python3 main.py]
```

## OpenCV
- Builing OpenCV from source: [here](https://docs.opencv.org/4.7.0/d2/de6/tutorial_py_setup_in_ubuntu.html)
- Check in [GPU Compute Capability](https://developer.nvidia.com/cuda-gpus)
```bash
python3 -c "import cv2; print(cv2.getBuildInformation())" # Check if OpenCV build information
python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())" # Check if CUDA is enabled

# CPU
docker build --progress=plain -t opencv-cpu -f OpenCV/dockerfiles/Dockerfile.cpu .
docker run --rm -it -v $(pwd):/workspace opencv-cpu python3 /workspace/OpenCV/example/main.py
docker run --rm -it -v $(pwd):/workspace opencv-cpu bash /workspace/OpenCV/example/run.sh

# CUDA
docker build --progress=plain -t opencv-cuda -f OpenCV/dockerfiles/Dockerfile.cuda .
docker run --rm -it --gpus all -v $(pwd):/workspace opencv-cuda python3 /workspace/OpenCV/example/main.py
docker run --rm -it --gpus all -v $(pwd):/workspace opencv-cuda bash /workspace/OpenCV/example/run.sh
```

<table>
  <tr>
    <td></td>
    <td colspan="2">CPU</td>
    <td colspan="2">CUDA</td>
  </tr>
  <tr>
    <td ></td>
    <td>Python</td>
    <td>C++</td>
    <td>Python</td>
    <td>C++</td>
  </tr>
  <tr>
    <td>FPS</td>
    <td>7.43</td>
    <td>7.86</td>
    <td>37.58</td>
    <td>58.17</td>
  </tr>
  <tr>
    <td>Preprocess (ms)</td>
    <td>2.358</td>
    <td>3.024</td>
    <td>2.213</td>
    <td>2.791</td>
  </tr>
  <tr>
    <td>Inference (ms)</td>
    <td>123.170</td>
    <td>121.077</td>
    <td>15.264</td>
    <td>11.529</td>
  </tr>
  <tr>
    <td>Postprocess (ms)</td>
    <td>9.129</td>
    <td>3.091</td>
    <td>9.131</td>
    <td>2.872</td>
  </tr>
</table>

## ONNXRuntime
```bash
# CPU
docker build --progress=plain -t onnxruntime-cpu -f ONNXRuntime/dockerfiles/Dockerfile.cpu .
docker run --rm -it -v $(pwd):/workspace onnxruntime-cpu python3 /workspace/ONNXRuntime/example/main.py
docker run --rm -it -v $(pwd):/workspace onnxruntime-cpu bash /workspace/ONNXRuntime/example/run.sh

# CUDA
docker build --progress=plain -t onnxruntime-cuda -f ONNXRuntime/dockerfiles/Dockerfile.cuda .
docker run --rm -it --gpus all -v $(pwd):/workspace onnxruntime-cuda python3 /workspace/ONNXRuntime/example/main.py
docker run --rm -it --gpus all -v $(pwd):/workspace onnxruntime-cuda bash /workspace/ONNXRuntime/example/run.sh

# TensorRT
docker build --progress=plain -t onnxruntime-tensorrt -f ONNXRuntime/dockerfiles/Dockerfile.tensorrt .
docker run --rm -it --gpus all -v $(pwd):/workspace onnxruntime-tensorrt python3 /workspace/ONNXRuntime/example/main.py
docker run --rm -it --gpus all -v $(pwd):/workspace onnxruntime-tensorrt bash /workspace/ONNXRuntime/example/run.sh

# OpenVINO (ONNXRuntime v1.6.3)
docker build --progress=plain -t onnxruntime-openvino -f ONNXRuntime/dockerfiles/Dockerfile.openvino .
docker run --rm -it -v $(pwd):/workspace onnxruntime-openvino python3 /workspace/ONNXRuntime/example/main.py
docker run --rm -it -v $(pwd):/workspace onnxruntime-openvino bash /workspace/ONNXRuntime/example/run.sh
```

<table>
  <tr>
    <td></td>
    <td colspan="2">CPU</td>
    <td colspan="2">CUDA</td>
    <td colspan="2">TensorRT</td>
  </tr>
  <tr>
    <td ></td>
    <td>Python</td>
    <td>C++</td>
    <td>Python</td>
    <td>C++</td>
    <td>Python</td>
    <td>C++</td>
  </tr>
  <tr>
    <td>FPS</td>
    <td>9.25</td>
    <td>10.36</td>
    <td>45.10</td>
    <td>62.12</td>
    <td>47.39</td>
    <td>60.68</td>
  </tr>
    <tr>
    <td>Preprocess (ms)</td>
    <td>2.470</td>
    <td>6.649</td>
    <td>2.123</td>
    <td>5.352</td>
    <td>2.080</td>
    <td>5.269</td>
  </tr>
  <tr>
    <td>Inference (ms)</td>
    <td>93.637</td>
    <td>86.349</td>
    <td>10.791</td>
    <td>7.822</td>
    <td>9.883</td>
    <td>8.362</td>
  </tr>
  <tr>
    <td>Postprocess (ms)</td>
    <td>11.991</td>
    <td>3.508</td>
    <td>9.260</td>
    <td>2.923</td>
    <td>9.140</td>
    <td>2.850</td>
  </tr>
</table>

## TensorRT
```bash
docker build --progress=plain -t tensorrt -f TensorRT/dockerfiles/Dockerfile .
docker run --rm -it --gpus all -v $(pwd):/workspace tensorrt python3 /workspace/TensorRT/example/main.py
docker run --rm -it --gpus all -v $(pwd):/workspace tensorrt bash /workspace/TensorRT/example/run.sh
```

<table>
  <tr>
    <td></td>
    <td colspan="2">TensorRT</td>
  </tr>
  <tr>
    <td ></td>
    <td>Python</td>
    <td>C++</td>
  </tr>
  <tr>
    <td>FPS</td>
    <td>50.35</td>
    <td>68.01</td>
  </tr>
  <tr>
    <td>Preprocess (ms)</td>
    <td>1.952</td>
    <td>5.602</td>
  </tr>
  <tr>
    <td>Inference (ms)</td>
    <td>8.882</td>
    <td>6.310</td>
  </tr>
  <tr>
    <td>Postprocess (ms)</td>
    <td>9.026</td>
    <td>2.791</td>
  </tr>
</table>


## Triton
```bash
docker build --progress=plain -t triton-server -f Triton/dockerfiles/Dockerfile.server .
docker build --progress=plain -t triton-client -f Triton/dockerfiles/Dockerfile.client .

mkdir -p Triton/model_repository/ensemble/1 Triton/model_repository/yolov8n/1
# docker run --rm -it --gpus all -v $(pwd):/workspace triton-server bash
/usr/src/tensorrt/bin/trtexec --onnx=/workspace/Assets/yolov8n.onnx --saveEngine=/workspace/Triton/model_repository/yolov8n/1/model.plan --explicitBatch

# Start Triton Inference Server
docker run --gpus=all -it --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/Triton/model_repository:/models triton-server tritonserver --model-repository=/models

# Run inference
docker run -it --rm --network host -v $(pwd):/workspace triton-client python3 /workspace/Triton/client.py
```

## ONNXRuntime and OpenCV on Intel MacOS
```bash
# Virtual environment
conda create -n cv python=3.8
conda activate cv
```

```bash
# OpenCV
bash osx_opencv.sh
cd OpenCV/example

## Python
python main.py

## C++: Export OPENCV_INSTALL_DIR
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$OPENCV_INSTALL_DIR
cmake --build build
./build/main
```

```bash
# ONNXRuntime CoreML
bash osx_ort_coreml.sh
cd ONNXRuntime/example

# Python
python main.py

# C++: Export OPENCV_INSTALL_DIR and ORT_INSTALL_DIR
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$OPENCV_INSTALL_DIR -DORT_INSTALL_DIR=$ORT_INSTALL_DIR
cmake --build build
./build/main
```
Note: Not all operators are supported in CoreML.
- `Resnet18.onnx` (1, 3, 640, 640) => CoreML CPU is 2x faster than CPU.
- `YOLOv8n.onnx` (1, 3, 640, 640) => CoreML CPU is 2x slower than CPU because of some unsupported operators.


Next:
- Update ONNXRuntime and OpenCV to newest version
- Try OpenCV with OpenVINO
- Optimize preprocessing and postprocessing in C++ version.