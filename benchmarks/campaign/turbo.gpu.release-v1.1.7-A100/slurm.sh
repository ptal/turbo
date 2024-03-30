#!/bin/bash -l
#SBATCH --time=20:00:00
#SBATCH --nodes=10
#SBATCH --partition=gpu
#SBATCH --account=p200244 
#SBATCH --qos=default 
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --output=slurm.out

echo $SLURM_JOB_NODELIST

./run.sh
