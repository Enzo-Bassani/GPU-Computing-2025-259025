cmake_minimum_required(VERSION 3.20)

project(gpu_project VERSION 1.0 LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)
# set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 70 72 75 80 86 87 89 90 100 101 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
set(CMAKE_CUDA_ARCHITECTURES 86)

include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB SOURCES
  ${PROJECT_SOURCE_DIR}/src/utils/*.cu
  ${PROJECT_SOURCE_DIR}/src/utils/*.cpp
)

add_executable(gpu_csr ${PROJECT_SOURCE_DIR}/src/program/gpu_csr.cu ${SOURCES})
target_link_libraries(gpu_csr PRIVATE common)
target_compile_options(gpu_csr PRIVATE -Wall -Wextra)  # Enable most warnings
target_compile_options(gpu_csr PRIVATE -O3)  # Optimization level

add_executable(gpu_naive ${PROJECT_SOURCE_DIR}/src/program/gpu_naive.cu ${SOURCES})
target_link_libraries(gpu_naive PRIVATE common)
target_compile_options(gpu_naive PRIVATE -Wall -Wextra)  # Enable most warnings
target_compile_options(gpu_naive PRIVATE -O3)  # Optimization level

add_executable(gpu_csr_stride ${PROJECT_SOURCE_DIR}/src/program/gpu_csr_stride.cu ${SOURCES})
target_link_libraries(gpu_csr_stride PRIVATE common)
target_compile_options(gpu_csr_stride PRIVATE -Wall -Wextra)  # Enable most warnings
target_compile_options(gpu_csr_stride PRIVATE -O3)  # Optimization level

add_executable(gpu_coo_add_atomic ${PROJECT_SOURCE_DIR}/src/program/gpu_coo_add_atomic.cu ${SOURCES})
target_link_libraries(gpu_coo_add_atomic PRIVATE common)
target_compile_options(gpu_coo_add_atomic PRIVATE -Wall -Wextra)  # Enable most warnings
target_compile_options(gpu_coo_add_atomic PRIVATE -O3)  # Optimization level

add_executable(gpu_coo_add_atomic_shared_mem ${PROJECT_SOURCE_DIR}/src/program/gpu_coo_add_atomic_shared_mem.cu ${SOURCES})
target_link_libraries(gpu_coo_add_atomic_shared_mem PRIVATE common)
target_compile_options(gpu_coo_add_atomic_shared_mem PRIVATE -Wall -Wextra)  # Enable most warnings
target_compile_options(gpu_coo_add_atomic_shared_mem PRIVATE -O3)  # Optimization level

add_executable(gpu_csr_constant_memory ${PROJECT_SOURCE_DIR}/src/program/gpu_csr_constant_memory.cu ${SOURCES})
target_link_libraries(gpu_csr_constant_memory PRIVATE common)
target_compile_options(gpu_csr_constant_memory PRIVATE -Wall -Wextra)  # Enable most warnings
target_compile_options(gpu_csr_constant_memory PRIVATE -O3)  # Optimization level
