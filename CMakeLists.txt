cmake_minimum_required(VERSION 2.8.4)
project(kalman)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(OCV_INC_DIR /usr/include/opencv/)
include_directories(${OCV_INC_DIR})

find_package(OpenCV 2.4 REQUIRED )

list(APPEND OpenCV_LIBRARIES opencv_nonfree)

set(SOURCE_FILES kalman.cpp kalman-test.cpp)
add_executable(kalman-test ${SOURCE_FILES})
target_link_libraries(kalman-test ${OpenCV_LIBRARIES})
