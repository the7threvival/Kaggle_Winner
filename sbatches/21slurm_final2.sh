#!/bin/bash
#SBATCH --time=360
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:32g:6
#SBATCH --output=21_final2.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nouafs@rpi.edu

srun \
    --job-name=run_21 \
    python train.py final 5800 3e-4 1 > 21final2.out 

