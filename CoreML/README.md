## macOS - M2 MacBook Air
```bash
conda create -yn coreml python=3.10 && conda activate coreml
pip install torch==2.5.0 coremltools==8.0 opencv-python~=4.10.0 pillow==11.0.0

# Install XCode from App Store
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer

# Python
python py/main.py

# Objective-C++ (OpenCV)
export OPENCV_INSTALL_DIR=/tmp/opencv
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$OPENCV_INSTALL_DIR
cmake --build build --config Release --parallel
SPDLOG_LEVEL=info ./build/main
```

<table>
  <tr>
    <td ></td>
    <td>Python</td>
    <td>Objective-C++</td>
  </tr>
  <tr>
    <td>FPS</td>
    <td>96.22</td>
    <td>179.99</td>
  </tr>
  <tr>
    <td>Preprocess</td>
    <td>1.301 ms</td>
    <td>2.184 ms</td>
  </tr>
  <tr>
    <td>Inference</td>
    <td>3.858 ms</td>
    <td>2.064 ms</td>
  </tr>
  <tr>
    <td>Postprocess</td>
    <td>5.234 ms</td>
    <td>1.308 ms</td>
  </tr>
</table>