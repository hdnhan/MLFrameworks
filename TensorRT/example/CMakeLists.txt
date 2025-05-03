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

find_package(CUDA REQUIRED)
message(STATUS "CUDA found: ${CUDA_VERSION_STRING} ${CUDA_INCLUDE_DIRS} ${CUDA_LIBRARIES}")

find_library(LIBNVINFER NAME nvinfer HINTS ${TENSORRT_LIB_PATH} REQUIRED)
find_library(LIBNVONNXPARSER NAME nvonnxparser HINTS ${TENSORRT_LIB_PATH} REQUIRED)
find_library(LIBNVINFER_PLUGIN NAME nvinfer_plugin HINTS ${TENSORRT_LIB_PATH} REQUIRED)
message(STATUS "TensorRT found: ${LIBNVINFER} ${LIBNVONNXPARSER} ${LIBNVINFER_PLUGIN}")

cuda_add_executable(main cpp/main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/cpp/base.cpp)
target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../../Common/cpp)
target_link_libraries(main ${OpenCV_LIBS} ${LIBNVINFER} ${LIBNVONNXPARSER} ${LIBNVINFER_PLUGIN} spdlog::spdlog cxxopts::cxxopts)
