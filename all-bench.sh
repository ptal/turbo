#!/bin/bash -l
#SBATCH --account=p200021 # project account
#SBATCH --partition=gpu                    # partition
#SBATCH --nodes=1                          # number of nodes
#SBATCH --ntasks=7                         # number of tasks
#SBATCH --ntasks-per-node=7                # number of tasks per node
#SBATCH --cpus-per-task=1                  # number of cores per task
#SBATCH --gpus-per-task=1                  # number of gpu per task
#SBATCH --time=0-12:00                     # time (DD-HH:MM)
#SBATCH -q default

srun -n 1 --exact python3 run-hpc.py 300 20 patterson &
srun -n 1 --exact python3 run-hpc.py 300 18 patterson &
srun -n 1 --exact python3 run-hpc.py 300 16 patterson &
srun -n 1 --exact python3 run-hpc.py 30 20 j30 &
srun -n 1 --exact python3 run-hpc.py 30 18 j30 &
srun -n 1 --exact python3 run-hpc.py 30 16 j30 &
srun -n 1 --exact python3 run-hpc.py 30 20 j60 &
wait
