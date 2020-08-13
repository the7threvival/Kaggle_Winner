#!/bin/bash
#SBATCH --time=360
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:32g:6
#SBATCH --output=23_2final.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nouafs@rpi.edu

srun \
    --job-name=run_23 \
    python train.py final2 8600 3e-4 1 > 232final.out 

