#!/bin/bash

#SBATCH --time=0-04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --job-name=udl
#SBATCH --output=udl-tv-base.out

module purge
module load Python/3.11.5-GCCcore-13.2.0

source $HOME/venvs/udl/bin/activate

python3 src/generate_scores.py

deactivate
