#!/bin/bash
#SBATCH --job-name=fragnet_broad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=fragnet_broad_%j.out
#SBATCH --error=fragnet_broad_%j.err

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
SCRIPT="${PROJECT_DIR}/fragnet_broad.py"

if [[ -d "${PROJECT_DIR}/FragNet" ]]; then
    FRAGNET_ROOT="${PROJECT_DIR}/FragNet"
else
    FRAGNET_ROOT="${PROJECT_DIR}/smiles_baseline2/FragNet"
fi

eval "$(conda shell.bash hook)"
conda activate fragnet

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

srun python -u "${SCRIPT}" \
    --input-path "${PROJECT_DIR}" \
    --fragnet-root "${FRAGNET_ROOT}" \
    --target-column log_P_upconversion \
    --split-method paper-scaffold \
    --n-trials 20 \
    --tuning-epochs 100 \
    --tuning-patience 20 \
    --prune-trials
