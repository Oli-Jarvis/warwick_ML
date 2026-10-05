#!/bin/bash
#SBATCH --job-name=fragnet_compare
#SBATCH --partition=nanosimd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=validate-%j.log
#SBATCH --error=validate-%j.log

set -euo pipefail
source /software/easybuild/software/Miniconda3/4.12.0/etc/profile.d/conda.sh
conda activate fragnet
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python -m unittest discover -s fragnet_workflow/tests -v
srun python -u fragnet_workflow/check_equivalence.py \
    --count 5 \
    --threads "$SLURM_CPUS_PER_TASK" \
    --output "validation_refactor_${SLURM_JOB_ID}"
