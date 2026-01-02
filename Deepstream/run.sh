#!/bin/bash
set -e

# Convert ONNX to TensorRT engine
if [ ! -f ../Triton/model_repository/yolov8n/1/model.plan ]; then
    trtexec --onnx=../Assets/yolov8n.onnx --saveEngine=../Triton/model_repository/yolov8n/1/model.plan
fi

# Build Triton postprocessing backend
export CUDA_ARCH_LIST=native
cmake -S ../Triton/model_repository/detection_postprocessing_cuda -B /tmp/postprocess_build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build /tmp/postprocess_build --parallel
cp /tmp/postprocess_build/libtriton_postprocess.so ../Triton/model_repository/detection_postprocessing_cuda/1

# Build DeepStream custom parser
cmake -S . -B /tmp/deepstream_build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build /tmp/deepstream_build --parallel
cp /tmp/deepstream_build/libnvds_infercustomparser.so .

deepstream-app -c pipeline.txt