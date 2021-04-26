cmake_minimum_required(VERSION 3.19)
project(kalman_filter_and_smoother)

set(CMAKE_CXX_STANDARD 14)

add_executable(kalman_filter_and_smoother src/main.cpp src/kalman_filter.cpp include/kalman_filter.h src/kalman_smoother.cpp include/kalman_smoother.h)

find_package (Eigen3 3.3 REQUIRED)
target_link_libraries (kalman_filter_and_smoother Eigen3::Eigen)