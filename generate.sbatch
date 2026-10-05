#!/bin/bash
#SBATCH --job-name=nmo_generate
#SBATCH --partition=nanosimd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=generate-%j.log
#SBATCH --error=generate-%j.log

set -euo pipefail

source /software/easybuild/software/Miniconda3/4.12.0/etc/profile.d/conda.sh
conda activate nmo

cd /storage/msszkb_grp/msshfg/fragnet_workflow

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

srun python -u start_fragnet_generation.py generate \
    --calls 256 \
    --seed 123 \
    --name pilot_seed123 \
    --threads "$SLURM_CPUS_PER_TASK" \
    --cpu

echo "Generation and projection completed: $(date)"
