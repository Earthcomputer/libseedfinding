cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(seedfinding_example LANGUAGES CXX)
find_package(libseedfinding)
add_executable(main main.cpp)
# improve compilation speed only in cmake 3.16+
target_precompile_headers(libseedfinding::libseedfinding INTERFACE "${LIBSEEDFINDING_HEADER_PATH}")
target_link_libraries(main libseedfinding::libseedfinding)