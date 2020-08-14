mkdir build
cd build
cmake ..
runas /user:Administrator "cmake --build . --config Release --target install"