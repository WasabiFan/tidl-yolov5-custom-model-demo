cmake_minimum_required(VERSION 3.1.0)

set(PROJ_NAME run_inference_images)
project(${PROJ_NAME})

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

set (CMAKE_CXX_STANDARD 17)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(SRCS
    run_inference_images.cpp)

add_executable(${PROJ_NAME} ${SRCS})
target_link_libraries( ${PROJ_NAME}
    ${OpenCV_LIBS}
    onnxruntime
    # Unclear why, but tivision_apps provides tivxMemAlloc
    tivision_apps
)

include_directories(
    /usr/include/onnxruntime/include
    # "core/session" dir is for "onnxruntime_c_api.h" and co., which other onnxruntime headers implicitly include
    /usr/include/onnxruntime/core/session

    /usr/include/processor_sdk/tiovx/include
)
