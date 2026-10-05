#!/bin/bash -l
#SBATCH --job-name=fragnet_embeddings
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=fragnet_embeddings_%j.out
#SBATCH --error=fragnet_embeddings_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fragnet

export PYTHONPATH="${SLURM_SUBMIT_DIR}/FragNet:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

test -f embeddings.py
test -f fragnet_log_P_upconversion_regression/experiment/ft.pt

srun python -u embeddings.py \
    --splits train \
    --batch-size 64 \
    --device cpu
