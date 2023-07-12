# TURBO

Turbo aims to be a constraint solver entirely on GPUs.
Its theoretical parallel model is described in this paper [AAAI 2022](http://hyc.io/papers/aaai2022.pdf).

### Dependencies

* Cuda compiler `nvcc` (>= 12.0).
* [libxml2](http://xmlsoft.org/)
* CMake version (>= 3.24)

The other dependencies will be pulled and compiled automatically by CMake.

### Configure, compile and run

The following command will configure and compile Turbo for GPU for the GPU architecture of your computer (native architecture).
The build time is rather slow (several hours), so to test, you might want to compile a "debug" version.

```
cmake --workflow --preset gpu-release --fresh
./build/gpu-release/turbo benchmarks/pat5.xml
```

Other compilation builds are also available:

* GPU Debug version: `cmake --workflow --preset gpu-debug --fresh`
* CPU Debug version: `cmake --workflow --preset cpu-debug --fresh`
* CPU Release version: `cmake --workflow --preset cpu-release --fresh`
```
