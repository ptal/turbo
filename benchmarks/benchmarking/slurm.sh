#!/bin/bash -l
#SBATCH --time=00:15:00
#SBATCH --nodes=2
#SBATCH --partition=gpu
#SBATCH --account=p200244 
#SBATCH --qos=default 
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL

echo $SLURM_JOB_NODELIST

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
