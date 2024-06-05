# TURBO

Turbo aims to be a constraint solver entirely on GPUs.
Its theoretical parallel model is described in this paper [AAAI 2022](http://hyc.io/papers/aaai2022.pdf).
Turbo is part of a larger project called [Lattice Land](https://github.com/lattice-land).
I write about Turbo in a [technical journal](https://lattice-land.github.io/1-turbo.html) where you can learn about the specifics of this solver.

### Dependencies

* Cuda compiler `nvcc` (>= 12.0).
* [libxml2](http://xmlsoft.org/)
* CMake (>= 3.27)
* Doxygen

The other dependencies will be pulled and compiled automatically by CMake.

### Configure, compile and run

You can first clone this repository, and then `git checkout` the latest released tag to ensure more stability.
The following command will configure and compile Turbo for GPU for the GPU architecture of your computer (native architecture).

```
cmake --workflow --preset gpu-release --fresh
./build/gpu-release/turbo -s -v -i -t 20000 benchmarks/example_wordpress7_500.fzn
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

### MiniZinc

The file [turbo.gpu.release.msc](https://github.com/ptal/turbo/blob/v1.1.0/benchmarks/minizinc/turbo.gpu.release.msc) can be copied in your Minizinc configuration directory (on Linux: `~/.minizinc/solvers`).
You should edit the file and update the paths of `executable` and `mznlib` to match the location of Turbo on your system.

You should be able to run Turbo directly from the [MiniZinc IDE](https://www.minizinc.org/) and from the command line:

```
minizinc -s -t 60000 --solver turbo.gpu.release benchmarks/mzn-challenge/2022/wordpress/Wordpress7_Offers500.dzn benchmarks/mzn-challenge/2022/wordpress/wordpress.mzn
```

### Developers

Please see [lattice-land](https://github.com/lattice-land/.github).
