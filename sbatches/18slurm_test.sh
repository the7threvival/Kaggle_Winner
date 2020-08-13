#!/bin/bash
#SBATCH --time=240
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --output=18_test.out
#SBATCH --gres=gpu:1

srun \
    --job-name=run_18 \
    python train.py small_test 0 3e-4 0 > 18test.out 

