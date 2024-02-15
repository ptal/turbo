#!/bin/bash

module load env/staging/2023.1
module load CUDA/12.2.0
module load libxml2/2.11.4-GCCcore-12.3.0
module load Python/3.11.3-GCCcore-12.3.0
module load parallel/20230722-GCCcore-12.3.0
# required to build Turbo, but not to run the xp.
module load CMake/3.26.3-GCCcore-12.3.0
module load Doxygen/1.9.7-GCCcore-12.3.0

export PATH=$PATH:/project/scratch/p200244/deps/libminizinc/build
source /project/scratch/p200244/lattice-land/turbo/benchmarks/pybench/bin/activate

