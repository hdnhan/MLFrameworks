FROM nvcr.io/nvidia/deepstream:8.0-triton-multiarch

RUN pip install opencv-python-headless "numpy<2.0" cmake
RUN apt-get update && apt-get install -y rapidjson-dev