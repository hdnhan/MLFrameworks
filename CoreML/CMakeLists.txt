cmake_minimum_required(VERSION 3.10)
project(CoreMLExample LANGUAGES CXX OBJCXX OBJC)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Optional
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra") # Optional

set(SRC src/bindings.mm src/bindings.h src/yolov8n.m src/yolov8n.h)
# add_library(bindings SHARED ${SRC})
# set_source_files_properties(${SRC} PROPERTIES COMPILE_FLAGS "-fobjc-arc")
# target_link_libraries (bindings PRIVATE "-framework CoreML -framework Foundation -framework Accelerate -framework CoreVideo")

# add_executable(main src/main.cpp)
# target_link_libraries(main PRIVATE bindings)

add_executable(main src/main.cpp ${SRC})
set_source_files_properties(${SRC} PROPERTIES COMPILE_FLAGS "-fobjc-arc")
target_link_libraries (main PRIVATE "-framework CoreML -framework Foundation -framework Accelerate -framework CoreVideo")
