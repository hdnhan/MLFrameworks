import os
import sys
import logging
import typing as T
from pathlib import Path

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

trt_logger = trt.Logger(trt.Logger.WARNING)

ROOT_DIR = Path(__file__).resolve().parents[3]

sys.path.append((ROOT_DIR / "Common/py").as_posix())
from base import Base  # noqa: E402

logger = logging.getLogger(__name__)


class TensorRT(Base):
    def __init__(
        self,
        model_name: str,
        dtype: T.Literal["f32", "f16", "i8"] = "f32",
    ) -> None:
        super().__init__()
        self.f_onnx = f"{model_name}.onnx"
        self.f_engine = f"{model_name}-{dtype}.engine"
        self.dtype = dtype

        trt.init_libnvinfer_plugins(trt_logger, "")
        self.engine = None

        if not os.path.exists(self.f_engine):
            assert os.path.exists(self.f_onnx)
            self.build()

        if not self.engine:
            self.load()
        self.context = self.engine.create_execution_context()

        # This allocates memory for network inputs/outputs on both CPU and GPU
        assert self.engine.num_io_tensors == 2
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
        """Runs the input image through the network.

        Args:
            image (np.ndarray): The input image (1, 3, h, w).

        Returns:
            np.ndarray: The output of the network (1, nc + 4, 8400).
        """
        # Transfer input data to the GPU.
        cuda.memcpy_htod(self.inputs, image.ravel())

        # Run inference.
        self.context.execute_v2(bindings=self.bindings)

        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh(self.outputs[0], self.outputs[1])

        return self.outputs[0].reshape(self.output_size)

    def load(self) -> None:
        runtime = trt.Runtime(trt_logger)
        with open(self.f_engine, "rb") as f:
            engine_data = f.read()
        self.engine = runtime.deserialize_cuda_engine(engine_data)

    def build(self) -> None:
        builder = trt.Builder(trt_logger)
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        if builder.platform_has_fast_fp16 and self.dtype == "f16":
            config.set_flag(trt.BuilderFlag.FP16)

        if builder.platform_has_fast_int8 and self.dtype == "i8":
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.INT8)

        flag = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, trt_logger)
        if not parser.parse_from_file(self.f_onnx):
            raise RuntimeError(f"failed to load ONNX file: {self.f_onnx}")

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        for inp in inputs:
            print(f'input "{inp.name}" with shape{inp.shape} {inp.dtype}')
        for out in outputs:
            print(f'output "{out.name}" with shape{out.shape} {out.dtype}')

        # Write file
        with builder.build_serialized_network(network, config) as plan, open(self.f_engine, "wb") as t:
            runtime = trt.Runtime(trt_logger)
            engine = runtime.deserialize_cuda_engine(plan)
            t.write(engine.serialize())


if __name__ == "__main__":
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    video_path = (ROOT_DIR / "Assets/video.mp4").as_posix()
    save_path = (ROOT_DIR / "Results/Linux-TensorRT-Python.mp4").as_posix()

    # Load the network
    session = TensorRT(f"{ROOT_DIR}/Assets/yolov8n")
    session.run(video_path, save_path)
