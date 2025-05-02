import os
import sys
import logging
from pathlib import Path

import numpy as np
import coremltools as ct

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append((ROOT_DIR / "Common/py").as_posix())
from base import Base  # noqa: E402

logger = logging.getLogger(__name__)


class CoreML(Base):
    def __init__(self) -> None:
        super().__init__()

        # Load the network
        self.model = ct.models.MLModel(f"{ROOT_DIR}/Assets/yolov8n.mlpackage")
        self.model.compute_units = ct.ComputeUnit.CPU_AND_NE

    def infer(self, image: np.ndarray) -> np.ndarray:
        """Runs the input image through the network.

        Args:
            image (np.ndarray): The input image (1, 3, h, w).

        Returns:
            np.ndarray: The output of the network (1, nc + 4, 8400).
        """
        input_name = self.model.get_spec().description.input[0].name
        output_name = self.model.get_spec().description.output[0].name
        return self.model.predict({input_name: image})[output_name]


if __name__ == "__main__":
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    video_path = (ROOT_DIR / "Assets/video.mp4").as_posix()
    logger.info(f"Video path: {video_path}")
    save_path = (ROOT_DIR / "Results/coreml-python.mp4").as_posix()
    coreml = CoreML()
    coreml.run(video_path, save_path)
