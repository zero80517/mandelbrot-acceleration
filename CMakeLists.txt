# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.18)
project ("mandelbrot-acceleration")

#set(ORIGINAL ON)
#set(OPENMP ON)
set(CUDA ON)

if(ORIGINAL)
    add_subdirectory ("original")
elseif(OPENMP)
    add_subdirectory ("openmp")
elseif(CUDA)
    add_subdirectory("cuda")
endif()

