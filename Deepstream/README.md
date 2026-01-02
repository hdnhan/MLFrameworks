```bash
docker build --progress=plain -t deepstream -f Dockerfile .

docker run -it --rm --gpus=all -v $(pwd)/../:/workspace -w /workspace/Deepstream deepstream bash run.sh
```
