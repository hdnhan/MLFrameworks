# Computer Vision Examples
Run inference on `yolov8n` using
- OpenCV: CPU and CUDA
- ONNXRuntime with different Execution Providers: CPU, CUDA, CoreML, OpenVINO, TensorRT
- TensorRT
- CoreML
- OpenVINO
- LibTorch/PyTorch: CPU, CUDA and MPS
- Triton Inference Server


## Generate required files
```bash
conda create -n export python=3.10
conda activate export
# Install required dependencies
pip install ultralytics coremltools onnx
# Generate ONNX model from PyTorch
cd path/to/CVExamples
cd Assets && python export.py
```


## Create a virtual environment (Native, not through Docker)
```bash
conda create -n cv python=3.10
conda activate cv
```


## Download video
```bash
cd path/to/CVExamples
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
All latency unit is _ms_

<table>
  <tr>
    <th></th>
    <th colspan="4">Linux (Docker + g4dn.xlarge) </th>
    <th colspan="2">macOS (Native)</th>
  </tr>

  <tr>
    <th></th>
    <th colspan="2">CPU</th>
    <th colspan="2">CUDA</th>
    <th colspan="2">CPU</th>
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
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Preprocess</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Inference</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Postprocess</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>

## ONNXRuntime
### Python
<table>
  <tr>
    <th></th>
    <th colspan="4">Linux (Docker + g4dn.xlarge) </th>
    <th colspan="2">macOS (Native)</th>
  </tr>
  <tr>
    <th></th>
    <th>CPU</th>
    <th>CUDA</th>
    <th>TensorRT</th>
    <th>OpenVINO</th>
    <th>CPU</th>
    <th>CoreML</th>
  </tr>
  <tr>
    <td>FPS</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
    <tr>
    <td>Preprocess</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Inference</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Postprocess</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>

### C++
<table>
  <tr>
    <th></th>
    <th colspan="4">Linux (Docker + g4dn.xlarge) </th>
    <th colspan="2">macOS (Native)</th>
  </tr>
  <tr>
    <th></th>
    <th>CPU</th>
    <th>CUDA</th>
    <th>TensorRT</th>
    <th>OpenVINO</th>
    <th>CPU</th>
    <th>CoreML</th>
  </tr>
  <tr>
    <td>FPS</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
    <tr>
    <td>Preprocess</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Inference</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Postprocess</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
</table>


## TensorRT
<table>
  <tr>
    <th></th>
    <th colspan="2">Linux (Docker + g4dn.xlarge) </th>
  </tr>
  <tr>
    <th></th>
    <th colspan="2">TensorRT</th>
  </tr>
  <tr>
    <td ></td>
    <td>Python</td>
    <td>C++</td>
  </tr>
  <tr>
    <td>FPS</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Preprocess</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Inference</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>Postprocess</td>
    <td></td>
    <td></td>
  </tr>
</table>


## Triton Inference Server
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
