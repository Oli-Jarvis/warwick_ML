#!/bin/bash

#SBATCH --job-name=tree_hyper_search
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=hyper_search_%j.out
#SBATCH --error=hyper_search_%j.err

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date)"

# Run from the directory where sbatch was submitted.
cd "$SLURM_SUBMIT_DIR"

# Run the combined-runs hyperparameter search.
srun python -u hyper_search3.py . \
    --threshold 0.5 \
    --positive-above \
    --output-dir hyper_ggs_0_5

echo "Finished: $(date)"
