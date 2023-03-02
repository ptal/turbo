#!/bin/sh

mkdir -p build/gpu-release
cmake -DCMAKE_BUILD_TYPE=Release -DGPU=ON -Bbuild/gpu-release &&
cmake --build build/gpu-release &&
cp build/gpu-release/turbo gturbo

