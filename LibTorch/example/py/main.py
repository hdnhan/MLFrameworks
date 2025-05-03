import os
import sys
import logging
from pathlib import Path

import torch
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[3]

sys.path.append((ROOT_DIR / "Common/py").as_posix())
from base import Base  # noqa: E402

logger = logging.getLogger(__name__)


class LibTorch(Base):
    def __init__(self, device: str) -> None:
        super().__init__()
        self.device = device
        self.model: torch.jit.ScriptModule = torch.jit.load(f"{ROOT_DIR}/Assets/yolov8n.torchscript")
        self.model.to(self.device)
        self.model.eval()

    def infer(self, image: np.ndarray) -> np.ndarray:
        """Runs the input image through the network.

        Args:
            image (np.ndarray): The input image (1, 3, h, w).

        Returns:
            np.ndarray: The output of the network (1, nc + 4, 8400).
        """
        with torch.no_grad():
            image = torch.from_numpy(image).to(self.device)
            output = self.model(image)
            output = output.cpu().numpy()
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using {device}")
    save_path = (ROOT_DIR / f"Results/{platform}-LibTorch-Python-{device}.mp4").as_posix()
    session = LibTorch(device=device)
    session.run(video_path, save_path)
