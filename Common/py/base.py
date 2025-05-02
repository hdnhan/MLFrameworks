import logging
import typing as T
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Base:
    def __init__(self) -> None:
        self.colors = {}

    def preprocess(self, image: np.ndarray, new_shape: T.Tuple[int, int]) -> np.ndarray:
        """Preprocess the input image.

        Args:
            image (np.ndarray): The input image (h, w, 3).
            new_shape (T.Tuple[int, int]): The new shape of the image (height, width).

        Returns:
            np.ndarray: The preprocessed image (1, 3, height, width).
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize but keep aspect ratio
        h, w = image.shape[:2]
        height, width = new_shape
        ratio = min(height / h, width / w)
        image = cv2.resize(image, (int(w * ratio), int(h * ratio)))
        # Pad to new_shape
        dh = (height - image.shape[0]) / 2
        dw = (width - image.shape[1]) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        image = image.transpose(2, 0, 1)
        image = image[np.newaxis, ...].astype(np.float32) / 255.0
        return image

    def infer(self, image: np.ndarray) -> np.ndarray:
        """Runs the input image through the network.

        Args:
            image (np.ndarray): The input image (1, 3, h, w).

        Returns:
            np.ndarray: The output of the network (1, nc + 4, 8400).
        """
        raise NotImplementedError

    def postprocess(
        self,
        output: np.ndarray,
        new_shape: T.Tuple[int, int],
        ori_shape: T.Tuple[int, int],
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Postprocess the output of the network.

        Args:
            output (np.ndarray): The output of the network (1, 84, 8400).
            new_shape (T.Tuple[int, int]): The new shape of the image (height, width).
            ori_shape (T.Tuple[int, int]): The original shape of the image (height, width).
            conf_thres (float, optional): The confidence threshold. Defaults to 0.25.
            iou_thres (float, optional): The IoU threshold. Defaults to 0.45.

        Returns:
            T.Tuple[np.ndarray, np.ndarray, np.ndarray]: The bounding boxes, scores, and class IDs.
        """
        assert output.shape == (1, 84, 8400)
        output = output[0]  # (84, 8400)
        output = output.transpose(1, 0)  # (8400, 84)

        bboxes = output[:, :4]  # (8400, 4) in cxcywh
        # cxcywh to xywh
        bboxes[..., 0] -= bboxes[..., 2] / 2
        bboxes[..., 1] -= bboxes[..., 3] / 2
        scores = np.max(output[:, 4:], axis=1)  # (8400,)
        class_ids = np.argmax(output[:, 4:], axis=1)  # (8400,)

        # Batched NMS
        keep = cv2.dnn.NMSBoxesBatched(
            bboxes=bboxes,  # type: ignore
            scores=scores,
            class_ids=class_ids,
            score_threshold=conf_thres,
            nms_threshold=iou_thres,
            top_k=300,
        )
        bboxes = bboxes[keep]  # type: ignore
        scores = scores[keep]
        class_ids = class_ids[keep]
        # xywh to xyxy
        bboxes[..., 2] += bboxes[..., 0]
        bboxes[..., 3] += bboxes[..., 1]

        # Scale and clip bboxes.
        bboxes = self.scale_boxes(bboxes, new_shape, ori_shape)
        return bboxes, scores, class_ids

    def scale_boxes(
        self,
        bboxes: np.ndarray,
        new_shape: T.Tuple[int, int],
        ori_shape: T.Tuple[int, int],
    ) -> np.ndarray:
        """Rescale bounding boxes to the original shape.

        Preprocess: ori_shape => new_shape
        Postprocess: new_shape => ori_shape

        Args:
            bboxes (np.ndarray): The bounding boxes in (x1, y1, x2, y2) format.
            new_shape (T.Tuple[int, int]): The new shape of the image (height, width).
            ori_shape (T.Tuple[int, int]): The original shape of the image (height, width).

        Returns:
            np.ndarray: The rescaled and clipped bounding boxes.
        """
        # calculate from ori_shape
        gain = min(new_shape[0] / ori_shape[0], new_shape[1] / ori_shape[1])  # gain  = old / new
        pad = (
            round((new_shape[1] - ori_shape[1] * gain) / 2 - 0.1),
            round((new_shape[0] - ori_shape[0] * gain) / 2 - 0.1),
        )  # wh padding

        bboxes[..., [0, 2]] -= pad[0]  # x padding
        bboxes[..., [1, 3]] -= pad[1]  # y padding

        bboxes /= gain
        # Clip bounding box coordinates to the image shape.
        bboxes[..., [0, 2]] = bboxes[..., [0, 2]].clip(0, ori_shape[1])  # x1, x2
        bboxes[..., [1, 3]] = bboxes[..., [1, 3]].clip(0, ori_shape[0])  # y1, y2
        return bboxes

    def draw(
        self,
        image: np.ndarray,
        bboxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
    ) -> None:
        """Draws the bounding boxes on the image.

        Args:
            image (np.ndarray): The input image (h, w, 3).
            bboxes (np.ndarray): The bounding boxes in (x1, y1, x2, y2) format.
            scores (np.ndarray): The scores of the bounding boxes.
            class_ids (np.ndarray): The class IDs of the bounding boxes.
        """
        for bbox, score, class_id in zip(bboxes, scores, class_ids):
            if class_id not in self.colors:
                self.colors[class_id] = np.random.randint(0, 256, size=3).tolist()
            color = self.colors[class_id]
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image,
                f"{score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

    def warmup(self) -> None:
        logger.debug("Warm up started.")
        image = np.random.rand(1, 3, 480, 640).astype(np.float32)
        new_shape = (640, 640)
        image = self.preprocess(image, new_shape)
        output = self.infer(image)
        bboxes, scores, class_ids = self.postprocess(output, new_shape, (480, 640))
        logger.debug("Warm up finished.")

    def run(self, video_path: str, save_path: str, verbose: bool = False) -> None:
        """Run the model on the video.

        Args:
            video_path (str): The path to the input video.
            save_path (str): The path to save the output video.
            verbose (bool, optional): Whether to print the FPS. Defaults to False.
        """
        logger.debug("Run started.")
        # Validate video path
        if not Path(video_path).is_file():
            raise FileNotFoundError(f"Cannot find video: {video_path}")
        # Validate save path
        if Path(save_path).exists():
            logger.warning(f"Save path already exists, overwriting: {save_path}")
        elif not Path(save_path).parent.is_dir():
            logger.info(f"Creating directory for save path: {save_path}")
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        else:
            logger.info(f"Save path: {save_path}")

        # Warmup
        for _ in range(10):
            self.warmup()

        # Load video
        cap: cv2.VideoCapture = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        writer: cv2.VideoWriter = cv2.VideoWriter()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        new_shape = (640, 640)  # (height, width)
        t_pres, t_infers, t_posts = 0, 0, 0
        count = 0
        while cap.isOpened():
            logger.debug(f"Frame {count}")
            ret, frame = cap.read()
            if not ret:
                break

            ori_shape = (frame.shape[0], frame.shape[1])

            # Preprocess
            start = perf_counter()
            image = self.preprocess(frame.copy(), new_shape)
            end = perf_counter()
            t_pre = end - start
            t_pres += t_pre
            logger.debug(f"Preprocess time: {t_pre / 1e3:.3f} ms")

            # Inference
            start = perf_counter()
            detections = self.infer(image)
            end = perf_counter()
            t_infer = end - start
            t_infers += t_infer
            logger.debug(f"Inference time: {t_infer / 1e3:.3f} ms")

            # Postprocess
            start = perf_counter()
            bboxes, scores, class_ids = self.postprocess(detections, new_shape, ori_shape)
            end = perf_counter()
            t_post = end - start
            t_posts += t_post
            logger.debug(f"Postprocess time: {t_post / 1e3:.3f} ms")

            fps = 1 / (t_pre + t_infer + t_post)
            logger.debug(f"FPS: {fps:.2f}")
            self.draw(frame, bboxes, scores, class_ids)
            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            count += 1

            if not writer.isOpened():
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    save_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
                    20,
                    (width, height),
                )
            writer.write(frame)
        logger.info(
            f"Preprocess: {t_pres * 1e3 / count:.3f} ms, Inference: {t_infers * 1e3 / count:.3f} ms, Postprocess: {t_posts * 1e3 / count:.3f} ms"
        )
        logger.info(f"FPS: {count / (t_pres + t_infers + t_posts):.2f}")

        if cap.isOpened():
            cap.release()
        if writer.isOpened():
            writer.release()
        logger.info("Run finished.")
