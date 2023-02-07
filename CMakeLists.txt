
cmake_minimum_required(VERSION 2.8)
add_definitions(-std=c++11)
project(image-stitching)
set(cmake_c_flags_debug "-g -Wall")

SET(OpenCV_DIR /usr/local/opencv-4.4/share/OpenCV)
find_package(OpenCV 4 REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        include
)

add_executable(${PROJECT_NAME}
        src/app.cc include/app.h
        src/sensor_data_interface.cc include/sensor_data_interface.h
        src/stitching_param_generater.cc include/stitching_param_generater.h
        src/image_stitcher.cc include/image_stitcher.h)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} -lpthread -lm)
