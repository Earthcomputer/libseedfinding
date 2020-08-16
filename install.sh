#!/usr/bin/env bash
mkdir -p build
# shellcheck disable=SC2164
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
sudo cmake --build . --target install