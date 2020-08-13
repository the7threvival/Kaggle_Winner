#!/bin/bash
#SBATCH --time=240
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --output=16_newWhale3.out
#SBATCH --gres=gpu:6
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nouafs@rpi.edu

srun \
    --job-name=run_16 \
    python train.py newWhale3 0 3e-4 0 > 16newWhale3.out 

