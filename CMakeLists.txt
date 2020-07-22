cmake_minimum_required(VERSION 3.13)
project(libseedfinding)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_library(libseedfinding dummy.cpp lcg.h)