import os
import typing as T
from time import perf_counter

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

logger = trt.Logger(trt.Logger.WARNING)


def preprocess(image: np.ndarray, new_shape: T.Tuple[int, int]) -> np.ndarray:
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


def postprocess(
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
    bboxes = scale_boxes(bboxes, new_shape, ori_shape)
    return bboxes, scores, class_ids


def scale_boxes(
    bboxes: np.ndarray, new_shape: T.Tuple[int, int], ori_shape: T.Tuple[int, int]
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
    gain = min(
        new_shape[0] / ori_shape[0], new_shape[1] / ori_shape[1]
    )  # gain  = old / new
    pad = round((new_shape[1] - ori_shape[1] * gain) / 2 - 0.1), round(
        (new_shape[0] - ori_shape[0] * gain) / 2 - 0.1
    )  # wh padding

    bboxes[..., [0, 2]] -= pad[0]  # x padding
    bboxes[..., [1, 3]] -= pad[1]  # y padding

    bboxes /= gain
    # Clip bounding box coordinates to the image shape.
    bboxes[..., [0, 2]] = bboxes[..., [0, 2]].clip(0, ori_shape[1])  # x1, x2
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]].clip(0, ori_shape[0])  # y1, y2
    return bboxes


colors = {}


def draw(
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
        if class_id not in colors:
            colors[class_id] = np.random.randint(0, 256, size=3).tolist()
        color = colors[class_id]
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


class TensorRTInfer:
    def __init__(
        self,
        model_name: str,
        dtype: T.Literal["f32", "f16", "i8"] = "f32",
    ) -> None:
        self.f_onnx = f"{model_name}.onnx"
        self.f_engine = f"{model_name}-{dtype}.engine"
        self.dtype = dtype

        trt.init_libnvinfer_plugins(logger, "")
        self.engine = None

        if not os.path.exists(self.f_engine):
            assert os.path.exists(self.f_onnx)
            self.build()

        if not self.engine:
            self.load()
        self.context = self.engine.create_execution_context()

        # This allocates memory for network inputs/outputs on both CPU and GPU
        assert self.engine.num_bindings == 2
        self.inputs, self.outputs, self.bindings = None, None, []
        self.input_size, self.output_size = (), ()
        for binding in self.engine:
            size = trt.volume(self.engine.get_tensor_shape(binding))
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
            # Allocate device buffers
            device_mem = cuda.mem_alloc(np.empty(size, dtype=dtype).nbytes)
            # Append the device buffer to device bindings.
            self.bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.inputs = device_mem  # device
                self.input_size = self.engine.get_tensor_shape(binding)
            else:
                self.outputs = [np.empty(size, dtype=dtype), device_mem]  # host, device
                self.output_size = self.engine.get_tensor_shape(binding)

    def infer(self, image: np.ndarray) -> np.ndarray:
        # Transfer input data to the GPU.
        cuda.memcpy_htod(self.inputs, image.ravel())

        # Run inference.
        self.context.execute_v2(bindings=self.bindings)

        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh(self.outputs[0], self.outputs[1])

        return self.outputs[0].reshape(self.output_size)

    def load(self) -> None:
        runtime = trt.Runtime(logger)
        with open(self.f_engine, "rb") as f:
            engine_data = f.read()
        self.engine = runtime.deserialize_cuda_engine(engine_data)

    def build(self) -> None:
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        # config.max_aux_streams = 8
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        if builder.platform_has_fast_fp16 and self.dtype == "f16":
            config.set_flag(trt.BuilderFlag.FP16)

        if builder.platform_has_fast_int8 and self.dtype == "i8":
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.INT8)

        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(self.f_onnx):
            raise RuntimeError(f"failed to load ONNX file: {self.f_onnx}")

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        for inp in inputs:
            print(f'input "{inp.name}" with shape{inp.shape} {inp.dtype}')
        for out in outputs:
            print(f'output "{out.name}" with shape{out.shape} {out.dtype}')

        # Write file
        with builder.build_serialized_network(network, config) as plan, open(
            self.f_engine, "wb"
        ) as t:
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(plan)
            t.write(engine.serialize())


def warmup(TRTSess: TensorRTInfer) -> None:
    image = np.random.rand(1, 3, 480, 640).astype(np.float32)
    new_shape = (640, 640)
    image = preprocess(image, new_shape)
    output = TRTSess.infer(image)
    bboxes, scores, class_ids = postprocess(output, new_shape, (480, 640))


def main(verbose: bool = False) -> None:
    # Load the network
    TRTSess = TensorRTInfer(model_name="/workspace/Assets/yolov8n")
    out_path = "/workspace/Results/tensorrt-python.mp4"

    # Warmup
    for _ in range(10):
        warmup(TRTSess)

    # Load video
    cap = cv2.VideoCapture("/workspace/Assets/video.mp4")
    out = None

    new_shape = (640, 640)  # (height, width)
    t_pres, t_infers, t_posts = 0, 0, 0
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        ori_shape = (frame.shape[0], frame.shape[1])

        # Preprocess
        start = perf_counter()
        image = preprocess(frame.copy(), new_shape)
        end = perf_counter()
        t_pre = end - start
        t_pres += t_pre

        # Inference
        start = perf_counter()
        detections = TRTSess.infer(image)
        end = perf_counter()
        t_infer = end - start
        t_infers += t_infer

        # Postprocess
        start = perf_counter()
        bboxes, scores, class_ids = postprocess(detections, new_shape, ori_shape)
        end = perf_counter()
        t_post = end - start
        t_posts += t_post

        fps = 1 / (t_pre + t_infer + t_post)
        draw(frame, bboxes, scores, class_ids)
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
        if verbose:
            print(
                f"{count} -> Preprocess: {t_pre * 1e3:.3f} ms, Inference: {t_infer * 1e3:.3f} ms, Postprocess: {t_post * 1e3:.3f} ms, FPS: {fps:.2f}"
            )

        if out is None:
            height, width = frame.shape[:2]
            out = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
                20,
                (width, height),
            )
        out.write(frame)
    print(
        f"Preprocess: {t_pres * 1e3 / count :.3f} ms, Inference: {t_infers * 1e3 / count :.3f} ms, Postprocess: {t_posts * 1e3 / count :.3f} ms"
    )
    print(f"FPS: {count / (t_pres + t_infers + t_posts):.2f}")


if __name__ == "__main__":
    main()
