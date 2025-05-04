## macOS - M2 Macbook Air - Native
```bash
conda create -yn libtorch python=3.10 && conda activate libtorch
pip install mnn opencv-python~=4.10.0
python3 -m MNN.tools.mnnconvert  --framework ONNX --modelFile ../Assets/yolov8n.onnx --MNNModel ../Assets/yolov8n.mnn --fp16 --bizCode MNN

# Python
python example/py/main.py

# C++
export OPENCV_INSTALL_DIR=/tmp/opencv
export MNN_INSTALL_DIR=/tmp/mnn
cmake -S example -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$OPENCV_INSTALL_DIR -DMNN_INSTALL_DIR=$MNN_INSTALL_DIR
cmake --build build --config Release --parallel
SPDLOG_LEVEL=debug ./build/main
```

<table>
  <tr>
    <td ></td>
    <td>Python</td>
    <td>C++</td>
  </tr>
  <tr>
    <td>FPS</td>
    <td>19.13</td>
    <td>74.16</td>
  </tr>
  <tr>
    <td>Preprocess</td>
    <td>2.156 ms</td>
    <td>1.593 ms</td>
  </tr>
  <tr>
    <td>Inference</td>
    <td>46.416 ms</td>
    <td>10.600 ms</td>
  </tr>
  <tr>
    <td>Postprocess</td>
    <td>3.696 ms</td>
    <td>1.289 ms</td>
  </tr>
</table>
