## Triton Inference Server - Docker
```bash
docker build --progress=plain -t triton-server -f dockerfiles/Dockerfile.server .
docker build --progress=plain -t triton-client -f dockerfiles/Dockerfile.client .

# Convert ONNX to TensorRT
mkdir -p model_repository/ensemble/1 model_repository/yolov8n/1
docker run --rm -it --gpus all -v $(pwd)/..:/workspace triton-server \
    /usr/src/tensorrt/bin/trtexec \
    --onnx=/workspace/Assets/yolov8n.onnx \
    --saveEngine=/workspace/Triton/model_repository/yolov8n/1/model.plan

# Start Triton Inference Server
docker run --gpus=all -it --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/model_repository:/models triton-server tritonserver --model-repository=/models

# Run inference
docker run -it --rm --network host -v $(pwd)/../:/workspace triton-client python3 /workspace/Triton/client.py
# Inference: 43.326 ms
# FPS: 23.08
```