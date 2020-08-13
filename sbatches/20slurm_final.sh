#!/bin/bash
#SBATCH --time=240
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:32g:2
#SBATCH --output=20_final.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nouafs@rpi.edu

srun \
    --job-name=run_20 \
    python train.py final 5800 3e-4 0 > 20final.out 

