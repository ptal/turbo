# TURBO

Turbo aims to be a constraint solver entirely on GPUs.

### Dependencies

* Cuda compiler `nvcc` (>= 11).
* [libxml2](http://xmlsoft.org/)
* CMake version (>= 3.20)

### Configure, compile and run

```
./config
cd build/gpu-release
make
cd ../..
./build/gpu-release/turbo benchmarks/pat5.xml
```