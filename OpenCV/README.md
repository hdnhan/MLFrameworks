## Linux - Docker
- Builing OpenCV from source: [here](https://docs.opencv.org/4.10.0/d2/de6/tutorial_py_setup_in_ubuntu.html)
- Check in [GPU Compute Capability](https://developer.nvidia.com/cuda-gpus)

```bash
python -c "import cv2; print(cv2.getBuildInformation())" # Check if OpenCV build information
python -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())" # Check if CUDA is enabled

# CPU
docker build --progress=plain -t opencv-cpu -f dockerfiles/Dockerfile.cpu .
docker run --rm -it -v $(pwd)/..:/workspace opencv-cpu python3 /workspace/OpenCV/example/py/main.py
docker run --rm -it -v $(pwd)/..:/workspace opencv-cpu bash /workspace/OpenCV/example/run.sh

# CUDA
docker build --progress=plain -t opencv-cuda -f dockerfiles/Dockerfile.cuda .
docker run --rm -it --gpus all -v $(pwd)/..:/workspace opencv-cuda python3 /workspace/OpenCV/example/py/main.py
docker run --rm -it --gpus all -v $(pwd)/..:/workspace opencv-cuda bash /workspace/OpenCV/example/run.sh
```

<table>
  <tr>
    <th></th>
    <th colspan="2">CPU</th>
    <th colspan="2">CUDA</th>
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
    <td>14.72</td>
    <td>15.66</td>
    <td>57.61</td>
    <td>71.92</td>
  </tr>
  <tr>
    <td>Preprocess</td>
    <td>1.880 ms</td>
    <td>2.014 ms</td>
    <td>2.256 ms</td>
    <td>6.180 ms</td>
  </tr>
  <tr>
    <td>Inference</td>
    <td>59.634 ms</td>
    <td>60.201 ms</td>
    <td>8.489 ms</td>
    <td>6.331 ms</td>
  </tr>
  <tr>
    <td>Postprocess</td>
    <td>6.400 ms</td>
    <td>1.631 ms</td>
    <td>6.612 ms</td>
    <td>1.394 ms</td>
  </tr>
</table>


## macOS - M2 Macbook Air - Docker
```bash
# CPU
docker build --progress=plain -t opencv-cpu -f dockerfiles/Dockerfile.cpu .
docker run --rm -it -v $(pwd)/..:/workspace opencv-cpu python3 /workspace/OpenCV/example/py/main.py
docker run --rm -it -v $(pwd)/..:/workspace opencv-cpu bash /workspace/OpenCV/example/run.sh
```

<table>
  <tr>
    <td ></td>
    <td>Python</td>
    <td>C++</td>
  </tr>
  <tr>
    <td>FPS</td>
    <td>16.61</td>
    <td>17.05</td>
  </tr>
  <tr>
    <td>Preprocess</td>
    <td>1.293 ms</td>
    <td>2.417 ms</td>
  </tr>
  <tr>
    <td>Inference</td>
    <td>52.915 ms</td>
    <td>55.154 ms</td>
  </tr>
  <tr>
    <td>Postprocess</td>
    <td>5.986 ms</td>
    <td>1.093 ms</td>
  </tr>
</table>


## macOS - M2 Macbook Air - Native
```bash
conda create -yn opencv python=3.10 && conda activate opencv

# Build OpenCV
bash build_opencv_osx.sh

# Python
python example/py/main.py

# C++
export OPENCV_INSTALL_DIR=/tmp/opencv
cmake -S example -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$OPENCV_INSTALL_DIR
cmake --build build --config Release
./build/main
```

<table>
  <tr>
    <td ></td>
    <td>Python</td>
    <td>C++</td>
  </tr>
  <tr>
    <td>FPS</td>
    <td>19.14</td>
    <td>22.28</td>
  </tr>
  <tr>
    <td>Preprocess</td>
    <td>2.267 ms</td>
    <td>2.135 ms</td>
  </tr>
  <tr>
    <td>Inference</td>
    <td>44.709 ms</td>
    <td>41.496 ms</td>
  </tr>
  <tr>
    <td>Postprocess</td>
    <td>5.257 ms</td>
    <td>1.254 ms</td>
  </tr>
</table>