cmake_minimum_required(VERSION 3.10)

project(cpu_project VERSION 1.0 LANGUAGES CXX)

add_executable(cpu_naive ${PROJECT_SOURCE_DIR}/src/program/cpu_naive.cpp)
target_link_libraries(cpu_naive PRIVATE common)
target_compile_options(cpu_naive PRIVATE -Wall -Wextra)  # Enable most warnings
target_compile_options(cpu_naive PRIVATE -O3 -g)  # Optimization level 3

add_executable(cpu_coo ${PROJECT_SOURCE_DIR}/src/program/cpu_coo.cpp)
target_link_libraries(cpu_coo PRIVATE common)
target_compile_options(cpu_coo PRIVATE -Wall -Wextra)  # Enable most warnings
target_compile_options(cpu_coo PRIVATE -O3 -g)  # Optimization level 3

add_executable(cpu_csr ${PROJECT_SOURCE_DIR}/src/program/cpu_csr.cpp)
target_link_libraries(cpu_csr PRIVATE common)
target_compile_options(cpu_csr PRIVATE -Wall -Wextra)  # Enable most warnings
target_compile_options(cpu_csr PRIVATE -O3 -g)  # Optimization level 3

add_executable(cpu_coo_struct ${PROJECT_SOURCE_DIR}/src/program/cpu_coo_struct.cpp)
target_link_libraries(cpu_coo_struct PRIVATE common)
target_compile_options(cpu_coo_struct PRIVATE -Wall -Wextra)  # Enable most warnings
target_compile_options(cpu_coo_struct PRIVATE -O3 -g)  # Optimization level 3
