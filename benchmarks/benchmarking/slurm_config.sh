#!/bin/bash -l
#SBATCH --nodes=1                          # number of nodes
#SBATCH --ntasks-per-node=4                # number of nodes
#SBATCH --gpus-per-task=1                  # number of gpu per task
#SBATCH --cpus-per-task=1                  # number of gpu per task
#SBATCH --partition=gpu                    # partition
#SBATCH --account=p200244                  # project account
#SBATCH --qos=default                      # SLURM qos

module load env/release/2023.1
module load CUDA/12.2.0
module load libxml2/2.11.4-GCCcore-12.3.0
module load Python/3.11.3-GCCcore-12.3.0

export PATH=$PATH:/project/scratch/p200244/deps/libminizinc/build
source /project/scratch/p200244/lattice-land/turbo/benchmarks/pybench/bin/activate

echo "Starting ${SLURM_ARRAY_TASK_ID}"

eval "srun -n 1 --exact python ${@:1} 0 &"
eval "srun -n 1 --exact python ${@:1} 1 &"
eval "srun -n 1 --exact python ${@:1} 2 &"
eval "srun -n 1 --exact python ${@:1} 3 &"
wait
