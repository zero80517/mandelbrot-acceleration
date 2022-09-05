cmake_minimum_required(VERSION 3.18)

project(cuda-library LANGUAGES CUDA CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

include(CheckLanguage)
check_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_library(cuda-library SHARED
    cudalibrary_global.h
    types.h
    cudalibrary.cpp
    cudalibrary.h
    kernel.cu
    kernel.cuh
)

target_compile_definitions(cuda-library PRIVATE CUDALIBRARY_LIBRARY)
