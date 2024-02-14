#!/bin/bash -l
#SBATCH --time=00:15:00
#SBATCH --nodes=2
#SBATCH --partition=gpu
#SBATCH --account=p200244 
#SBATCH --qos=default 
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --output=slurm.out

echo $SLURM_JOB_NODELIST

./run.sh
