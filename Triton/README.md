## Triton Inference Server - Docker
```bash
docker build --progress=plain -t triton-server -f dockerfiles/Dockerfile.server .
docker build --progress=plain -t triton-client -f dockerfiles/Dockerfile.client .

# Start Triton Inference Server
docker run -it --rm --gpus=all -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/../:/workspace -w /workspace/Triton triton-server bash run.sh

# Run inference
docker run -it --rm --network host -v $(pwd)/../:/workspace triton-client python3 /workspace/Triton/client.py

# Inference: 43.326 ms
# FPS: 23.08
```