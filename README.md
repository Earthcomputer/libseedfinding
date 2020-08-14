LibSeedFinding

# Install

After installation with Cmake, a find_package(libseedfinding) is available.
This creates a libseedfinding::libseedfinding target (if found).
It can be linked like so:

`target_link_libraries(your_exe libseedfinding::libseedfinding)`

The following will build & install for later use.

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