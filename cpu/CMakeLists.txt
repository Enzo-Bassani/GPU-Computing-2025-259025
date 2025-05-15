# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Project name and version
project(MyProject VERSION 1.0 LANGUAGES C)

# Generate compile commands to use with Clang
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Set the C++ standard to C++17 (you can change this to C++11, C++14, etc.)
# set(CMAKE_CXX_STANDARD 21)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

# Specify the include directories (you can add more paths if needed)
include_directories(${PROJECT_SOURCE_DIR}/include)

# Example: Add a subdirectory for more complex projects with multiple components
# add_subdirectory(src)  # Uncomment if your project has source files in a separate directory

# This section handles adding all source files dynamically using GLOB.
# You can manually list sources or use file() to automatically collect them.

# Example: Automatically collect all .cpp files in the src directory
file(GLOB SOURCES
  ${PROJECT_SOURCE_DIR}/src/utils/*.c
  # ${PROJECT_SOURCE_DIR}/src/**/*.c
)

# Alternatively, you can specify the source files manually like so:
# set(SOURCES src/main.cpp src/utils.cpp)

# Define the main executable
add_executable(cpu_naive ${PROJECT_SOURCE_DIR}/src/program/matrixMul_cpu_naive.c ${SOURCES})
# Example: GCC/Clang-specific flags
target_compile_options(cpu_naive PRIVATE -Wall -Wextra)  # Enable most warnings
target_compile_options(cpu_naive PRIVATE -O3)  # Optimization level 3


# Define the main executable
add_executable(cpu_coo ${PROJECT_SOURCE_DIR}/src/program/matrixMul_cpu_coo.c ${SOURCES})
# Example: GCC/Clang-specific flags
target_compile_options(cpu_coo PRIVATE -Wall -Wextra)  # Enable most warnings
target_compile_options(cpu_coo PRIVATE -O3)  # Optimization level 3

# Define the main executable
add_executable(cpu_csr ${PROJECT_SOURCE_DIR}/src/program/cpu_csr.c ${SOURCES})
# Example: GCC/Clang-specific flags
target_compile_options(cpu_csr PRIVATE -Wall -Wextra)  # Enable most warnings
target_compile_options(cpu_csr PRIVATE -O3)  # Optimization level 3

# Optionally, link libraries to your project (modify as per your needs)
# You can link to system libraries or third-party libraries (e.g., Boost, OpenGL, etc.)

# Example: Linking system libraries
# target_link_libraries(cpu_naive PRIVATE pthread)  # On Linux for multi-threading support

# Example: Linking third-party libraries (e.g., Boost)
# find_package(Boost 1.70 REQUIRED)
# target_link_libraries(cpu_naive PRIVATE Boost::Boost)

# Libraries can also be linked without the find_package method. For example, for the math library
# target_link_libraries(cpu_naive PRIVATE m)

# Set the installation rules (optional)
# This allows you to install the built binaries, libraries, and headers to a specific location.
# For example, to install the executable to /usr/local/bin
# install(TARGETS cpu_naive DESTINATION bin)

# If you have headers to install, you can do it like this:
# install(DIRECTORY include/ DESTINATION include)
