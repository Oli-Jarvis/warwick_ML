#!/bin/bash
#SBATCH --job-name=fragnet_fine
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=fragnet_fine_%j.out
#SBATCH --error=fragnet_fine_%j.err

# The broad job ran without an explicit partition. If Vulcan later requires
# one, remove one leading # below and replace the placeholder.
##SBATCH --partition=YOUR_GPU_PARTITION

set -euo pipefail

# Submit from /storage/msszkb_grp/msshfg, where fragnet_fine.py and the broad
# output directory are located.
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
TRAINING_SCRIPT="${PROJECT_DIR}/fragnet_fine.py"
BROAD_DIR="${PROJECT_DIR}/fragnet_combined_log_P_upconversion_broad_optuna"

if [[ -d "${PROJECT_DIR}/FragNet" ]]; then
    FRAGNET_ROOT="${PROJECT_DIR}/FragNet"
elif [[ -d "${PROJECT_DIR}/smiles_baseline2/FragNet" ]]; then
    FRAGNET_ROOT="${PROJECT_DIR}/smiles_baseline2/FragNet"
else
    echo "ERROR: Could not find FragNet or smiles_baseline2/FragNet." >&2
    exit 1
fi

required_paths=(
    "${TRAINING_SCRIPT}"
    "${BROAD_DIR}/fragnet_broad_optimised.yaml"
    "${BROAD_DIR}/broad_optuna/best_broad_hyperparameters.json"
    "${BROAD_DIR}/graph_data/train.pkl"
    "${BROAD_DIR}/graph_data/val.pkl"
    "${BROAD_DIR}/graph_data/test.pkl"
)

for required_path in "${required_paths[@]}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "ERROR: Required path not found: ${required_path}" >&2
        exit 1
    fi
done

# Initialise Conda in the non-interactive SLURM shell.
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: Conda could not be initialised." >&2
    exit 1
fi
conda activate fragnet

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONPATH="${FRAGNET_ROOT}:${PYTHONPATH:-}"

echo "Job ID:          ${SLURM_JOB_ID}"
echo "Node:            $(hostname)"
echo "Project:         ${PROJECT_DIR}"
echo "FragNet:         ${FRAGNET_ROOT}"
echo "Broad results:   ${BROAD_DIR}"
echo "Fine trials:     50 total (persistent and resumable)"
echo "Final test run:  disabled"
python --version
nvidia-smi || true

# This performs only validation-based model selection. If the 48-hour walltime
# is reached, submit this same file again: the persistent Optuna database makes
# --n-trials the desired total, rather than 50 additional trials.
srun python -u "${TRAINING_SCRIPT}" \
    --broad-dir "${BROAD_DIR}" \
    --fragnet-root "${FRAGNET_ROOT}" \
    --n-trials 50 \
    --tuning-epochs 100 \
    --tuning-patience 20 \
    --prune-trials \
    --confirm-top-candidates \
    --confirm-top-k 3 \
    --confirmation-seeds 123 456 789

echo "Fine search completed successfully."
echo "The held-out test set has not been evaluated."
