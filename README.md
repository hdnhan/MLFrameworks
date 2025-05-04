# Computer Vision Examples
Run inference on `yolov8n` using
- OpenCV: CPU and CUDA
- ONNXRuntime: CPU, CUDA, CoreML, OpenVINO, TensorRT
- TensorRT
- CoreML: CPU, GPU and Neural Engine
- LibTorch: CPU, CUDA and MPS
- OpenVINO: CPU and GPU
- MNN: CPU, CUDA, Metal, CoreML and TensorRT
- Triton Inference Server


## Generate required files
```bash
conda create -yn export python=3.10 && conda activate export
pip install -r requirements.txt
cd Assets && python export.py && cd ..
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
