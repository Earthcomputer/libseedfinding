cmake_minimum_required(VERSION 3.13)
project(libseedfinding)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

add_subdirectory(src)
add_subdirectory(test)

target_link_libraries(libseedfinding)