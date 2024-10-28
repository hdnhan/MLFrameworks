import os
import torch
from ultralytics import YOLO
import coremltools as ct
import numpy as np

device = "cpu"
model = YOLO("yolov8n.pt")

print("\nONNX...")
model.export(format="onnx", device=device, opset=20)  # 12 OpenCV DNN 4.7.0

print("\nTorchScript...")
if torch.backends.mps.is_available():
    model.export(format="torchscript", device="mps")
    trace = torch.jit.load("yolov8n.torchscript")
    trace = torch.jit.freeze(trace)
    torch.jit.save(trace, "yolov8n_mps.torchscript")

model.export(format="torchscript", device=device)
trace = torch.jit.load("yolov8n.torchscript")
trace = torch.jit.freeze(trace)
torch.jit.save(trace, "yolov8n.torchscript")

print("\nCoreML...")
# model.export(format="coreml", device=device)
m = ct.convert(
    torch.jit.load("yolov8n.torchscript"),
    convert_to="mlprogram",
    inputs=[ct.TensorType(name="image", shape=(1, 3, 640, 640), dtype=np.float32)],
    outputs=[ct.TensorType(name="output", dtype=np.float32)],
    compute_units=ct.ComputeUnit.ALL,
    compute_precision=ct.precision.FLOAT16,
)
m.save("yolov8n.mlpackage")

try:
    os.system("mpsgraphtool convert -onnx yolov8n.onnx -packageName yolov8n -path .")
except Exception as e:
    print(f"Error: {e}")

try:
    os.system("xcrun coremlc compile yolov8n.mlpackage .")
except Exception as e:
    print(f"Error: {e}")
