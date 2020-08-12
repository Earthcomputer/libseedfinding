cmake_minimum_required(VERSION 3.17)
project(libseedfinding CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)

find_package(Python COMPONENTS Interpreter Development)
if (Python_FOUND)
    execute_process(COMMAND python ${CMAKE_HOME_DIRECTORY}/cmake/gen_version.py)
else ()
    error("You are missing Python the version.h was not generated")
endif ()

include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA support")
else ()
    message(STATUS "No CUDA support")
endif ()

add_subdirectory(src)
add_subdirectory(test)

target_link_libraries(libseedfinding)
