## Linux - Docker
```bash
docker build --progress=plain -t tensorrt -f dockerfiles/Dockerfile .
docker run --rm -it --gpus all -v $(pwd)/..:/workspace tensorrt python3 /workspace/TensorRT/example/py/main.py
docker run --rm -it --gpus all -v $(pwd)/..:/workspace tensorrt bash /workspace/TensorRT/example/run.sh
```

<table>
  <tr>
    <td ></td>
    <td>Python</td>
    <td>C++</td>
  </tr>
  <tr>
    <td>FPS</td>
    <td>106.72</td>
    <td></td>
  </tr>
  <tr>
    <td>Preprocess</td>
    <td>1.995 ms</td>
    <td>1.764 ms</td>
  </tr>
  <tr>
    <td>Inference</td>
    <td>2.275 ms</td>
    <td>1.649 ms</td>
  </tr>
  <tr>
    <td>Postprocess</td>
    <td>5.101 ms</td>
    <td>1.388 ms</td>
  </tr>
</table>