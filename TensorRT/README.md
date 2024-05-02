
## Linux - Docker
```bash
cd path/CVExamples/TensorRT

docker build --progress=plain -t tensorrt -f dockerfiles/Dockerfile .
docker run --rm -it --gpus all -v $(pwd)/..:/workspace tensorrt python3 /workspace/TensorRT/example/py/main.py
docker run --rm -it --gpus all -v $(pwd)/..:/workspace tensorrt bash /workspace/TensorRT/example/run.sh
```