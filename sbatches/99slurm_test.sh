#!/bin/bash
#SBATCH --time=240
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=99_test.out

srun \
    --job-name=run_test \
    python test.py > 99test.out 

