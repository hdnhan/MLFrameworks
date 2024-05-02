import torch
from ultralytics import YOLO

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model = YOLO("yolov8n.pt")

print("\nONNX...")
model.export(format="onnx", device=device, opset=12) # 12 OpenCV DNN 4.7.0

print("\nTorchScript...")
model.export(format="torchscript", device=device)
trace = torch.jit.load("yolov8n.torchscript")
trace = torch.jit.freeze(trace)
torch.jit.save(trace, "yolov8n.torchscript")

print("\nCoreML...")
model.export(format="coreml", device=device)
