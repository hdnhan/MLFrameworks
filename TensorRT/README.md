
## Linux - Ubuntu
```bash
docker build --progress=plain -t tensorrt -f dockerfiles/Dockerfile .
docker run --rm -it --gpus all -v $(pwd)/..:/workspace tensorrt python3 /workspace/TensorRT/example/main.py
docker run --rm -it --gpus all -v $(pwd)/..:/workspace tensorrt bash /workspace/TensorRT/example/run.sh
```