#!/bin/bash
#SBATCH --time=240
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --output=sbatch5.out
#SBATCH --gres=gpu:6
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nouafs@rpi.edu

srun \
    --job-name=run_5 \
    python train.py flukes5 > flukes5.out 
