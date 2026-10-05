#!/bin/bash
#SBATCH --job-name=nmo_project
#SBATCH --partition=nanosimd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=project-%j.log
#SBATCH --error=project-%j.log

set -euo pipefail

source /software/easybuild/software/Miniconda3/4.12.0/etc/profile.d/conda.sh
conda activate fragnet

cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}/fragnet_workflow"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

srun python -u project_molecules.py \
    --name "${1:-pilot_seed123}" \
    --threads "$SLURM_CPUS_PER_TASK"

echo "Projection completed: $(date)"
