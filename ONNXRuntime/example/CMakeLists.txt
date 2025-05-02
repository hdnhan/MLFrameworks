cmake_minimum_required(VERSION 3.5)
project(ORTExample LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Optional
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra") # Optional

# CMAKE MODULES
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/../../cmake/")
include(CPM)

# EXTERNAL LIBRARIES
cpmaddpackage("gh:gabime/spdlog#v1.15.2")
cpmaddpackage("gh:jarro2783/cxxopts#v3.2.0")

find_package(OpenCV REQUIRED) # Need `CMAKE_PREFIX_PATH=$OPENCV_INSTALL_DIR`
# find_package(onnxruntime REQUIRED) # not working
find_library(ORT_LIBRARY NAME onnxruntime HINTS ${ORT_INSTALL_DIR}/lib REQUIRED)
set(ORT_INCLUDE_DIRS ${ORT_INSTALL_DIR}/include/onnxruntime)

find_package(CUDA)
if(CUDA_FOUND)
  message(STATUS "CUDA found: ${CUDA_VERSION_STRING} ${CUDA_INCLUDE_DIRS} ${CUDA_LIBRARIES}")
  add_compile_definitions(USE_CUDA)
else()
  message(STATUS "CUDA not found")
endif()

if(EXISTS ${ORT_INSTALL_DIR}/include/onnxruntime/openvino_provider_factory.h)
  message(STATUS "OpenVINO found")
  add_compile_definitions(USE_OPENVINO)
else()
  message(STATUS "OpenVINO not found")
endif()

if(EXISTS ${ORT_INSTALL_DIR}/include/onnxruntime/coreml_provider_factory.h)
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
else()
  message(STATUS "TensorRT not found")
endif()

add_executable(main cpp/main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/cpp/base.cpp)
target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS} ${ORT_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/cpp)
target_link_libraries(main ${OpenCV_LIBS} ${ORT_LIBRARY} spdlog::spdlog cxxopts::cxxopts)
