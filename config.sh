#!/bin/sh

mkdir -p build/cpu-debug
mkdir -p build/cpu-ldebug
mkdir -p build/cpu-release
mkdir -p build/gpu-debug
mkdir -p build/gpu-ldebug
mkdir -p build/gpu-release

cmake -DCMAKE_BUILD_TYPE=Debug -DGPU=OFF -Bbuild/cpu-debug
cmake -DCMAKE_BUILD_TYPE=LDebug -DGPU=OFF -Bbuild/cpu-ldebug
cmake -DCMAKE_BUILD_TYPE=Release -DGPU=OFF -Bbuild/cpu-release
cmake -DCMAKE_BUILD_TYPE=Debug -DGPU=ON -Bbuild/gpu-debug
cmake -DCMAKE_BUILD_TYPE=LDebug -DGPU=ON -Bbuild/gpu-ldebug
cmake -DCMAKE_BUILD_TYPE=Release -DGPU=ON -Bbuild/gpu-release