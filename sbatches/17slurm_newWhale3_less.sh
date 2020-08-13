#!/bin/bash
#SBATCH --time=240
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --output=17_newWhale3_less.out
#SBATCH --gres=gpu:6
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nouafs@rpi.edu

srun \
    --job-name=run_17 \
    python train.py newWhale3_less 0 3e-4 0 > 17newWhale3_less.out 

