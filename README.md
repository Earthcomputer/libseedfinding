# LibSeedFinding

# Grand public usage

It's recommended to use the example repository here: https://github.com/hube12/seedfinding_example.

It's solely a cmake file that configure and add the library as a local library so the executable can be link against.
 
The main part are the cmake/libseedfinding.cmake and the following lines :

```cmake
##################### Configure the library ##############################
# find the external project and include it
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(seedfinding)
# create the library and bind the cmake script to it, the library is available everywhere now
add_library(libseedfinding INTERFACE)
add_dependencies(libseedfinding seedfinding)
target_include_directories(libseedfinding INTERFACE ${libseedfinding_INCLUDE_DIR})
##########################################################################
```

# Install in the system

After installation (you can use the install.bat as administrator, or the install.sh with sudo privileges) with Cmake, 
a find_package(libseedfinding) is available.

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

See example/ after installing

# Development

Create new .h in include and add it to the CmakeLists.txt after the line 177:

```cmake
######################### LIST of all header files to include ############################
list(APPEND LIBSEEDFINDING_HEADERS
        "lcg.h"
        "simplex.h"
        "util.h"
        "version.h")
if (LIBSEEDFINDING_IS_USING_CUDA)
    list(APPEND LIBSEEDFINDING_HEADERS
            "kernels.cuh")
endif()
##########################################################################################
```


# Bugs and untested

So far the standalone static library is not tested, see split.py
