cmake_minimum_required(VERSION 3.17)
project(libseedfinding)

set(CMAKE_CUDA_STANDARD 14)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA support")
else()
    message(STATUS "No CUDA support")
endif()

add_subdirectory(src)
add_subdirectory(test)

target_link_libraries(libseedfinding)