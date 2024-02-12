#!/bin/bash -l
#SBATCH --nodes=5                          # number of nodes
#SBATCH --partition=gpu                    # partition
#SBATCH --account=p200244                  # project account
#SBATCH --qos=default                      # SLURM qos
#SBATCH --export=ALL

module load env/release/2023.1
module load CUDA/12.2.0
module load libxml2/2.11.4-GCCcore-12.3.0
module load Python/3.11.3-GCCcore-12.3.0
# required to build Turbo, but not to run the xp.
module load CMake/3.26.3-GCCcore-12.3.0
module load Doxygen/1.9.7-GCCcore-12.3.0

export PATH=$PATH:/project/scratch/p200244/deps/libminizinc/build
source /project/scratch/p200244/lattice-land/turbo/benchmarks/pybench/bin/activate

./run.sh
