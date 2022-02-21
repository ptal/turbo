#!/bin/bash -l
#SBATCH --account=p200021 # project account
#SBATCH --partition=gpu                    # partition
#SBATCH --nodes=1                          # number of nodes
#SBATCH --ntasks=2                         # number of tasks
#SBATCH --ntasks-per-node=2                # number of tasks per node
#SBATCH --cpus-per-task=1                  # number of cores per task
#SBATCH --gpus-per-task=1                  # number of gpu per task
#SBATCH --time=0-00:10                     # time (DD-HH:MM)
#SBATCH -q test

srun -n 1 --exact python3 run-hpc.py 18 patterson &
srun -n 1 --exact python3 run-hpc.py 16 patterson &
wait
