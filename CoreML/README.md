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
cmake --build build --config Release
./build/main
```

<table>
  <tr>
    <td ></td>
    <td>Python</td>
    <td>Objtive-C++</td>
  </tr>
  <tr>
    <td>FPS</td>
    <td>60.47</td>
    <td>165.21</td>
  </tr>
  <tr>
    <td>Preprocess</td>
    <td>2.491 ms</td>
    <td>2.283 ms</td>
  </tr>
  <tr>
    <td>Inference</td>
    <td>8.458 ms</td>
    <td>2.400 ms</td>
  </tr>
  <tr>
    <td>Postprocess</td>
    <td>5.589 ms</td>
    <td>1.370 ms</td>
  </tr>
</table>