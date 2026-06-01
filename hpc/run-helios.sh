#!/bin/bash -l
#SBATCH --time=00:15:00
#SBATCH -p gpu
#SBATCH -A tutorial
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 4 GPUs so 4 tasks per nodes.
#SBATCH --mem=0
#SBATCH --qos=normal
#SBATCH --export=ALL
#SBATCH --output=slurm-wordpress-turbo-gpu.out

# Exits when an error occurs.
set -e
set -x # useful for debugging.

source $1

#cd ~/lattice-land/turbo/build/gpu-release-local
#ptxas -arch=sm_90 -v --suppress-stack-size-warning --allow-expensive-optimizations false -O0  "turbo.ptx"  -o "turbo.cubin"
#cmake --workflow --preset gpu-release-local --fresh

cd ~/lattice-land/turbo
./build/gpu-release-local/turbo -s -t 60000 -and 256 -or 132 -arch hybrid benchmarks/example_wordpress7_500.fzn

exit 0

