#!/bin/sh

mkdir -p build/gpu-debug
cmake -DCMAKE_BUILD_TYPE=Debug -DGPU=ON -Bbuild/gpu-debug &&
cmake --build build/gpu-debug &&
cp build/gpu-debug/turbo gturbo

