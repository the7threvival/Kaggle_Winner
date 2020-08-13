#!/bin/bash
#SBATCH --time=60
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:32g:2
#SBATCH --output=19_newWhale3_less.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nouafs@rpi.edu

srun \
    --job-name=run_19 \
    python train.py newWhale3_less 0 3e-4 0 > 19newWhale3_less.out 

