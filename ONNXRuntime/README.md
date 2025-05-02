## Linux - Docker
```bash
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

<table>
  <tr>
    <th></th>
    <th colspan="2">CPU</th>
    <th colspan="2">CUDA</th>
    <th colspan="2">TensorRT</th>
    <th colspan="2">OpenVINO</th>
  </tr>
  <tr>
    <td ></td>
    <td>Python</td>
    <td>C++</td>
    <td>Python</td>
    <td>C++</td>
    <td>Python</td>
    <td>C++</td>
    <td>Python</td>
    <td>C++</td>
  </tr>
  <tr>
    <td>FPS</td>
    <td>15.71</td>
    <td>16.37</td>
    <td>75.79</td>
    <td>109.99</td>
    <td>106.00</td>
    <td>178.39</td>
    <td>17.80</td>
    <td>19.46</td>
  </tr>
  <tr>
    <td>Preprocess</td>
    <td>2.467 ms</td>
    <td>5.602 ms</td>
    <td>1.845 ms</td>
    <td>2.839 ms</td>
    <td>1.846 ms</td>
    <td>1.813 ms</td>
    <td>1.980 ms</td>
    <td>2.387 ms</td>
  </tr>
  <tr>
    <td>Inference</td>
    <td>54.422 ms</td>
    <td>53.941 ms</td>
    <td>4.854 ms</td>
    <td>4.708 ms</td>
    <td>2.476 ms</td>
    <td>2.354 ms</td>
    <td>47.239 ms</td>
    <td>47.385 ms</td>
  </tr>
  <tr>
    <td>Postprocess</td>
    <td>6.776 ms</td>
    <td>1.531 ms</td>
    <td>6.494 ms</td>
    <td>1.544 ms</td>
    <td>5.112 ms</td>
    <td>1.439 ms</td>
    <td>6.966 ms</td>
    <td>1.626 ms</td>
  </tr>
</table>


## macOS - M2 Macbook Air - Docker
```bash
# CPU
docker build --progress=plain -t onnxruntime-cpu -f dockerfiles/Dockerfile.cpu .
docker run --rm -it -v $(pwd)/..:/workspace onnxruntime-cpu python3 /workspace/ONNXRuntime/example/py/main.py
docker run --rm -it -v $(pwd)/..:/workspace onnxruntime-cpu bash /workspace/ONNXRuntime/example/run.sh
```
<table>
  <tr>
    <td ></td>
    <td>Python</td>
    <td>C++</td>
  </tr>
  <tr>
    <td>FPS</td>
    <td>12.36</td>
    <td>15.79</td>
  </tr>
  <tr>
    <td>Preprocess</td>
    <td>1.452 ms</td>
    <td>2.709 ms</td>
  </tr>
  <tr>
    <td>Inference</td>
    <td>71.284 ms</td>
    <td>58.690 ms</td>
  </tr>
  <tr>
    <td>Postprocess</td>
    <td>8.178 ms</td>
    <td>1.936 ms</td>
  </tr>
</table>


## macOS - M2 Macbook Air - Native
```bash
conda create -yn onnxruntime python=3.10 && conda activate onnxruntime

# Build OpenCV
bash build_onnxruntime_osx.sh

# Python
python example/py/main.py

# C++
export OPENCV_INSTALL_DIR=/tmp/opencv
export ORT_INSTALL_DIR=/tmp/onnxruntime
cmake -S example -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$OPENCV_INSTALL_DIR -DORT_INSTALL_DIR=$ORT_INSTALL_DIR
cmake --build build --config Release --parallel
SPDLOG_LEVEL=debug ./build/main
```

<table>
  <tr>
    <th></th>
    <th colspan="2">CPU</th>
    <th colspan="2">CoreML</th>
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
    <td>22.00</td>
    <td>25.28</td>
    <td>46.97</td>
    <td>63.68</td>
  </tr>
  <tr>
    <td>Preprocess</td>
    <td>2.673 ms</td>
    <td>2.311 ms</td>
    <td>2.320 ms</td>
    <td>2.249 ms</td>
  </tr>
  <tr>
    <td>Inference</td>
    <td>37.020 ms</td>
    <td>35.892 ms</td>
    <td>13.198 ms</td>
    <td>12.032 ms</td>
  </tr>
  <tr>
    <td>Postprocess</td>
    <td>5.763 ms</td>
    <td>1.358 ms</td>
    <td>5.772 ms</td>
    <td>1.422 ms</td>
  </tr>
</table>