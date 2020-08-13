#!/bin/bash
#SBATCH --time=240
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:32g:2
#SBATCH --output=69_embed_test.out

srun \
    --job-name=run_embed_test \
    python embed_test.py > 69embed_test.out 

