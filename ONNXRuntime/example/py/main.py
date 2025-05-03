import os
import sys
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT_DIR = Path(__file__).resolve().parents[3]

sys.path.append((ROOT_DIR / "Common/py").as_posix())
from base import Base  # noqa: E402

logger = logging.getLogger(__name__)


class ONNXRuntime(Base):
    def __init__(self, ep: str) -> None:
        super().__init__()

        options = ort.SessionOptions()
        options.enable_profiling = False
        provider_options = {}
        if ep == "TensorrtExecutionProvider":
            provider_options = {
                "device_id": 0,
                "trt_max_workspace_size": 3 * 1024 * 1024 * 1024,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": f"{ROOT_DIR}/Assets",
            }
        elif ep == "CoreMLExecutionProvider":
            provider_options = {
                "coreml_flags": 0,
            }

        self.sess = ort.InferenceSession(
            f"{ROOT_DIR}/Assets/yolov8n.onnx",
            options,
            providers=[ep],
            provider_options=[provider_options],
        )

    def infer(self, image: np.ndarray) -> np.ndarray:
        """Runs the input image through the network.

        Args:
            image (np.ndarray): The input image (1, 3, h, w).

        Returns:
            np.ndarray: The output of the network (1, nc + 4, 8400).
        """
        return self.sess.run(None, {self.sess.get_inputs()[0].name: image})[0]


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

    eps = ort.get_available_providers()
    for ep in eps:
        logger.info(f"Using {ep}")
        save_path = (ROOT_DIR / f"Results/{platform}-ONNXRuntime-Python-{ep}.mp4").as_posix()
        session = ONNXRuntime(ep)
        session.run(video_path, save_path)
