```bash
docker build --progress=plain -t deepstream -f Dockerfile .

docker run -it --rm --gpus=all -v $(pwd)/../:/workspace -w /workspace/Deepstream deepstream bash run.sh
```

Note:
- Python `detection_preprocessing` in Triton can be easily replaced by `preprocess` in `config_infer.txt` for better performance.
- To use gRPC, change `triton` in `config_infer.txt`.
- See more examples in `/opt/nvidia/deepstream/deepstream/samples/configs`