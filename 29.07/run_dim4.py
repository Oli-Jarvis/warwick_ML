#!/bin/bash -l
#SBATCH --job-name=dim4_pca
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=dim4_%j.out
#SBATCH --error=dim4_%j.err

set -euo pipefail

# Submit from the 29.07 directory containing dim4.py, smiles_baseline2/,
# ggs_2/, and GA_best_candidates_full_history.csv.
cd "${SLURM_SUBMIT_DIR}"

source "$(conda info --base)/etc/profile.d/conda.sh"

# Defaults are suitable for the Warwick setup. Override at submission time with,
# for example: sbatch --export=ALL,CONDA_ENV=chemplot,PCA_COMPONENTS=20 run_dim4_cpu.sh
CONDA_ENV="${CONDA_ENV:-fragnet}"
PCA_COMPONENTS="${PCA_COMPONENTS:-10}"

conda activate "${CONDA_ENV}"

export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

test -f dim4.py
test -d smiles_baseline2
test -d ggs_2
test -f GA_best_candidates_full_history.csv

echo "Conda environment: ${CONDA_ENV}"
echo "PCA components: ${PCA_COMPONENTS}"
echo "Working directory: ${SLURM_SUBMIT_DIR}"

srun python -u dim4.py . \
    --pca-components "${PCA_COMPONENTS}" \
    --output-dir dim_plots
