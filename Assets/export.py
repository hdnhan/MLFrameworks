import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # for macOS
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="onnx", opset=12)  # 12 for OpenCV
model.export(format="coreml")
