cmake_minimum_required(VERSION 3.28)
project(DenseKAN C)

set(CMAKE_C_STANDARD 11)

add_executable(DenseKAN main.c
        splines.c
        splines.h
        kan_layer.c
        kan_layer.h
)
