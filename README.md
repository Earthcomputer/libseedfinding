# LibSeedFinding

# Grand public usage

It's recommended to use the example repository here: https://github.com/hube12/seedfinding_example which is solely a cmake file that configure and add the library as 
a local library so the executable can be link against, the main part are the cmake/libseedfinding.cmake and the line 
```cmake
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(libseedfinding)
add_dependencies(YOUR_TARGET libseedfinding)
target_include_directories(YOUR_TARGET PUBLIC ${libseedfinding_INCLUDE_DIR})
```

# Install

After installation with Cmake, a find_package(libseedfinding) is available.
This creates a libseedfinding::libseedfinding target (if found).
It can be linked like so:

`target_link_libraries(your_exe libseedfinding::libseedfinding)`

The following will build & install for later use. (you can also do `cmake . && make install_libseedfinding`)

Linux/macOS:
```
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
sudo cmake --build . --target install
```
Windows:
```
mkdir build
cd build
cmake ..
runas /user:Administrator "cmake --build . --config Release --target install"
```

# Usage

See example/

# Development

Create new .h in include and add them in the CmakeLists.txt in the install part as well as in the in config


# Bugs and untested

So far the cuda build is not tested

So far the standalone executable is not tested
