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

export GST_DEBUG=0
mkdir -p ../Results

# Run DeepStream application
deepstream-app -c pipeline.txt

# Alternatively, run the DeepStream pipeline using Python
python3 pipeline.py

# Equivalent GStreamer command (args: -v for verbose, -e for EOS on exit)
gst-launch-1.0 -e \
    filesrc location=../Assets/video.mp4 ! \
    qtdemux ! h264parse ! nvv4l2decoder ! queue ! mux.sink_0 \
    nvstreammux name=mux batch-size=1 width=1920 height=1080 batched-push-timeout=40000 live-source=0 ! \
    nvinferserver config-file-path=config_infer.txt ! \
    nvvideoconvert ! \
    nvmultistreamtiler rows=1 columns=1 width=1920 height=1080 ! \
    nvdsosd ! \
    nvvideoconvert ! \
    nvv4l2h264enc bitrate=2000000 ! \
    h264parse ! \
    qtmux ! \
    filesink location=../Results/deepstream_commandline.mp4 sync=0
