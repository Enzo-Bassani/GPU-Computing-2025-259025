cmake_minimum_required(VERSION 3.20)

project(CuProject LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 23)
# set(CMAKE_CUDA_STANDARD 23)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(common)
add_subdirectory(cpu)
add_subdirectory(gpu)
