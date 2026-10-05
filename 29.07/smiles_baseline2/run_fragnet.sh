#!/bin/bash
#SBATCH --job-name=fragnet
#SBATCH --partition=compute
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=fragnet_%j.out
#SBATCH --error=fragnet_%j.err

source /software/easybuild/software/Miniconda3/4.12.0/etc/profile.d/conda.sh
conda activate fragnet_cpu

export PYTHONPATH=$PWD/FragNet:$PYTHONPATH

python test_frag.py
