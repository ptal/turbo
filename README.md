# TURBO

Turbo aims to be a constraint solver entirely on GPUs.
Its theoretical parallel model is described in this paper [AAAI 2022](http://hyc.io/papers/aaai2022.pdf).
Turbo is part of a larger project called [Lattice Land](https://github.com/lattice-land).
I have started to write about Turbo in a [technical journal](https://lattice-land.github.io/1-turbo.html) where you can learn details about this solver.

### Dependencies

* Cuda compiler `nvcc` (>= 12.0).
* [libxml2](http://xmlsoft.org/)
* CMake version (>= 3.24)

The other dependencies will be pulled and compiled automatically by CMake.

### Configure, compile and run

You can first clone this repository, and then `git checkout` the latest released tag to ensure more stability.
The following command will configure and compile Turbo for GPU for the GPU architecture of your computer (native architecture).

```
cmake --workflow --preset gpu-release --fresh
./build/gpu-release/turbo -v -s benchmarks/data/patterson.task-rd.fzn/pat2.fzn
```

Other compilation builds are also available:

* GPU Debug version: `cmake --workflow --preset gpu-debug --fresh`
* CPU Debug version: `cmake --workflow --preset cpu-debug --fresh`
* CPU Release version: `cmake --workflow --preset cpu-release --fresh`

Alternatively, you can use these commands without presets and workflow (it is useful compilation scenario falling outside what is provided by the presets):

```
cmake -DCMAKE_BUILD_TYPE=Release -DGPU=ON -DREDUCE_PTX_SIZE=ON -DCMAKE_VERBOSE_MAKEFILE=ON -Bbuild/gpu-release
cmake --build build/gpu-release
```

### Developers

Please see [lattice-land](https://github.com/lattice-land/.github).
