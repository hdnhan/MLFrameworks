cmake_minimum_required(VERSION 3.5)
project(ORTExample LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Optional
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra") # Optional

find_package(OpenCV REQUIRED) # Need `CMAKE_PREFIX_PATH=$OPENCV_INSTALL_DIR`
# find_package(onnxruntime REQUIRED) # not working
find_library(ORT_LIBRARY NAME onnxruntime HINTS ${ORT_INSTALL_DIR}/lib REQUIRED)
set(ORT_INCLUDE_DIRS ${ORT_INSTALL_DIR}/include/onnxruntime/core/session) # older version

find_package(CUDA)
if(CUDA_FOUND)
  message(STATUS "CUDA found: ${CUDA_VERSION_STRING} ${CUDA_INCLUDE_DIRS} ${CUDA_LIBRARIES}")
  add_compile_definitions(USE_CUDA)
else()
  message(STATUS "CUDA not found")
endif()

if(EXISTS ${ORT_INSTALL_DIR}/include/onnxruntime/openvino_provider_factory.h) # v1.16.3
  list(APPEND ORT_INCLUDE_DIRS ${ORT_INSTALL_DIR}/include/onnxruntime) # v1.16.3
  message(STATUS "OpenVINO found")
  add_compile_definitions(USE_OPENVINO)
else()
  message(STATUS "OpenVINO not found")
endif()

if(EXISTS ${ORT_INSTALL_DIR}/include/onnxruntime/coreml_provider_factory.h) # v1.16.3
  list(APPEND ORT_INCLUDE_DIRS ${ORT_INSTALL_DIR}/include/onnxruntime) # v1.16.3
  message(STATUS "CoreML found")
  add_compile_definitions(USE_COREML)
else()
  message(STATUS "CoreML not found")
endif()

find_library(LIBNVINFER NAME nvinfer HINTS ${TENSORRT_LIB_PATH})
find_library(LIBNVONNXPARSER NAME nvonnxparser HINTS ${TENSORRT_LIB_PATH})
find_library(LIBNVINFER_PLUGIN NAME nvinfer_plugin HINTS ${TENSORRT_LIB_PATH})
if(LIBNVINFER AND LIBNVONNXPARSER AND LIBNVINFER_PLUGIN)
  message(STATUS "TensorRT found: ${LIBNVINFER} ${LIBNVONNXPARSER} ${LIBNVINFER_PLUGIN}")
  add_compile_definitions(USE_TENSORRT)
  list(APPEND ORT_INCLUDE_DIRS ${ORT_INSTALL_DIR}/include/onnxruntime/core/providers/tensorrt)
else()
  message(STATUS "TensorRT not found")
endif()

add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS} ${ORT_LIBRARY})
target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS} ${ORT_INCLUDE_DIRS})
