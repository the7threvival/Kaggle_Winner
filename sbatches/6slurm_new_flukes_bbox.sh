#!/bin/bash
#SBATCH --time=240
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:32g:2
#SBATCH --output=6_new_flukes_bbox.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nouafs@rpi.edu

srun \
    --job-name=run_6 \
    python train.py new_flukes 0 3e-4 0 > 6new_flukes_bbox.out 

