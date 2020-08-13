#!/bin/bash
#SBATCH --time=240
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=70_test.out

srun \
    --job-name=run_test \
    python test.py > 70test.out 

