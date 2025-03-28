#!/bin/bash -l
#SBATCH --time=03:00:00
#SBATCH -p gpu
#SBATCH -A tutorial
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 4 GPUs so 4 tasks per nodes.
#SBATCH --mem=0
#SBATCH --qos=normal
#SBATCH --export=ALL
#SBATCH --output=slurm-compileturbo-gpu.out

# Exits when an error occurs.
set -e
set -x # useful for debugging.

source $1

#cd ~/lattice-land/turbo/build/gpu-release-local
#ptxas -arch=sm_90 -v --suppress-stack-size-warning --allow-expensive-optimizations false -O0  "turbo.ptx"  -o "turbo.cubin"
cmake --workflow --preset gpu-release-local --fresh
exit 0

