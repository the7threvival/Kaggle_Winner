#!/bin/bash
#SBATCH --time=240
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --output=8sbatch_new_flukes_SGD.out
#SBATCH --gres=gpu:6
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nouafs@rpi.edu

srun \
    --job-name=run_8 \
    python train.py new_flukes 0 3e-3 False > 8new_flukes_SGD.out 

