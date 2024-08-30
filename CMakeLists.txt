cmake_minimum_required(VERSION 3.28)
project(DenseKAN LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 11)

set(CUDA_SOURCES
        src/CUDA/kan_layer.cu
)

set(C_SOURCES
        src/main.c
        src/KAN/kan_layer.c
        src/KAN/splines.c
)

add_executable(DenseKAN ${C_SOURCES} ${CUDA_SOURCES})


target_include_directories(DenseKAN PRIVATE
        src/KAN
        src/CUDA
)