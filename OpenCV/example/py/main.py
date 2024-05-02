import sys
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[3]

sys.path.append((ROOT_DIR / "Common/py").as_posix())
from base import Base


class OpenCV(Base):
    def __init__(self, use_cuda: bool = False) -> None:
        super().__init__()
        self.model: cv2.dnn.Net = cv2.dnn.readNet(f"{ROOT_DIR}/Assets/yolov8n.onnx")
        if use_cuda:
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def infer(self, image: np.ndarray) -> np.ndarray:
        """Runs the input image through the network.

        Args:
            image (np.ndarray): The input image (1, 3, h, w).

        Returns:
            np.ndarray: The output of the network (1, nc + 4, 8400).
        """
        # Sets the input to the network.
        self.model.setInput(image)

        # Runs the forward pass to get output of the output layers.
        layers = self.model.getUnconnectedOutLayersNames()
        outputs = self.model.forward(layers)
        return outputs[0]


if __name__ == "__main__":
    if sys.platform.startswith("linux"):
        platform = "Linux"
    elif sys.platform == "darwin":
        platform = "macOS"
    else:
        raise RuntimeError(f"Unknown platform: {sys.platform}")

    video_path = (ROOT_DIR / "Assets/video.mp4").as_posix()

    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        print("Using CUDA")
        save_path = (ROOT_DIR / f"Results/{platform}-OpenCV-Python-CUDA.mp4").as_posix()
        session = OpenCV(use_cuda=True)
        session.run(video_path, save_path)

    print("Using CPU")
    save_path = (ROOT_DIR / f"Results/{platform}-OpenCV-Python-CPU.mp4").as_posix()
    session = OpenCV()
    session.run(video_path, save_path)
