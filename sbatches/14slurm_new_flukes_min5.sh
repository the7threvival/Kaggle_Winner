#!/bin/bash
#SBATCH --time=240
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --output=14_new_flukes_min5.out
#SBATCH --gres=gpu:6
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nouafs@rpi.edu

srun \
    --job-name=run_14 \
    python train.py new_flukes5 0 3e-4 0 > 14new_flukes_min5.out 

