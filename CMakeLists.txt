cmake_minimum_required(VERSION 3.18)
project(single_pass_apose LANGUAGES CUDA CXX)

# Language standards
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to generating code for commonly available SM versions.
# Override with -DCMAKE_CUDA_ARCHITECTURES="80;86" etc.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90)
endif()

# Main test/benchmark executable
add_executable(apose_test src/main.cu)
target_include_directories(apose_test PRIVATE src)
set_target_properties(apose_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Optional: suppress verbose ptxas output unless requested
if(NOT DEFINED VERBOSE_PTX)
  target_compile_options(apose_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
endif()
