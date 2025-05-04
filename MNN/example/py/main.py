import os
import sys
import logging
from pathlib import Path

import MNN
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[3]

sys.path.append((ROOT_DIR / "Common/py").as_posix())
from base import Base  # noqa: E402

logger = logging.getLogger(__name__)


# https://github.com/alibaba/MNN/blob/master/pymnn/examples/MNNEngineDemo/mobilenet_demo.py
class PyMNN(Base):
    def __init__(self) -> None:
        super().__init__()
        model_path = (ROOT_DIR / "Assets/yolov8n.mnn").as_posix()
        logger.info(f"Model path: {model_path}")

        self.interpreter = MNN.Interpreter(model_path)
        config = {}
        config["backend"] = "METAL"
        self.session = self.interpreter.createSession(config)
        logger.info(f"Run on backendtype: {self.interpreter.getSessionInfo(self.session, 2)}")

        self.input_tensor = self.interpreter.getSessionInput(self.session)
        self.output_tensor = self.interpreter.getSessionOutput(self.session)
        logger.info(f"Input shape: {self.input_tensor.getShape()}")
        logger.info(f"Output shape: {self.output_tensor.getShape()}")
        # Create output placeholder on host
        self.output_host_tensor = MNN.Tensor(
            self.output_tensor.getShape(),
            self.output_tensor.getDataType(),
            np.zeros(self.output_tensor.getShape(), dtype=np.float32),
            self.output_tensor.getDimensionType(),
        )

    def infer(self, image: np.ndarray) -> np.ndarray:
        """Runs the input image through the network.

        Args:
            image (np.ndarray): The input image (1, 3, h, w).

        Returns:
            np.ndarray: The output of the network (1, nc + 4, 8400).
        """
        assert image.shape == self.input_tensor.getShape()
        tmp_input = MNN.Tensor(
            self.input_tensor.getShape(),
            self.input_tensor.getDataType(),
            image,
            self.input_tensor.getDimensionType(),
        )
        self.input_tensor.copyFrom(tmp_input)
        self.interpreter.runSession(self.session)
        self.output_tensor.copyToHostTensor(self.output_host_tensor)
        output = np.array(self.output_host_tensor.getData()).reshape(self.output_host_tensor.getShape())
        return output


if __name__ == "__main__":
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if sys.platform.startswith("linux"):
        platform = "Linux"
    elif sys.platform == "darwin":
        platform = "macOS"
    else:
        raise RuntimeError(f"Unknown platform: {sys.platform}")

    video_path = (ROOT_DIR / "Assets/video.mp4").as_posix()
    logger.info(f"Video path: {video_path}")

    save_path = (ROOT_DIR / f"Results/{platform}-MNN-Python.mp4").as_posix()
    session = PyMNN()
    session.run(video_path, save_path)
