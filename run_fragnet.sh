#!/bin/bash
#SBATCH --job-name=fragnet_combined
#SBATCH --chdir=/storage/msszkb_grp/msshfg
#SBATCH --output=/storage/msszkb_grp/msshfg/fragnet_combined_%j.out
#SBATCH --error=/storage/msszkb_grp/msshfg/fragnet_combined_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

# Submit with: sbatch run_fragnet_combined.sh
# No GPU request and no CUDA module or nvidia-smi dependency.
# Uses the site's default partition; override at submission if required:
# sbatch --partition=YOUR_CPU_PARTITION run_fragnet_combined.sh
# Resubmitting resumes the last complete epoch or cached graph preparation.

set -euo pipefail

readonly PROJECT_DIR=/storage/msszkb_grp/msshfg
readonly FRAGNET_ENV=/home/chem/msshfg/.conda/envs/fragnet
readonly FRAGNET_PYTHON="$FRAGNET_ENV/bin/python"
readonly TRAINING_SCRIPT="$PROJECT_DIR/run_fragnet.py"

# Select the exact environment interpreter, regardless of the submitting
# shell's activated environment. Conda itself is not needed to launch it.
export PATH="$FRAGNET_ENV/bin:$PATH"
export CONDA_PREFIX="$FRAGNET_ENV"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
unset PYTHONPATH PYTHONHOME

# Force CPU execution even on a host with an NVIDIA driver installed.
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg

cd "$PROJECT_DIR"
if [[ ! -x "$FRAGNET_PYTHON" ]]; then
    echo "ERROR: fragnet Python not found: $FRAGNET_PYTHON" >&2
    exit 1
fi
if [[ ! -f "$TRAINING_SCRIPT" ]]; then
    echo "ERROR: Training script not found: $TRAINING_SCRIPT" >&2
    exit 1
fi

printf 'Job: %s\nHost: %s\nPython: %s\nCPU threads: %s\n' \
    "${SLURM_JOB_ID:-manual}" "$(hostname)" "$FRAGNET_PYTHON" "$OMP_NUM_THREADS"

# Verify the actual interpreter and native dependencies before doing any work.
"$FRAGNET_PYTHON" - <<'PY'
import os
import sys
from pathlib import Path

expected = Path(os.environ['CONDA_PREFIX']).resolve()
if Path(sys.prefix).resolve() != expected:
    raise RuntimeError(f'Wrong Python environment: {sys.prefix}; expected {expected}')

import numpy
import pandas
import yaml
import networkx
import scipy
import sklearn
import lmdb
import tqdm
import rdkit
import torch
import torch_geometric
import torch_scatter

root = Path('/storage/msszkb_grp/msshfg/smiles_baseline2/FragNet')
if not (root / 'fragnet/model/gat/gat2.py').is_file():
    raise FileNotFoundError(f'FragNet repository missing: {root}')
sys.path.insert(0, str(root))
from fragnet.dataset.data import CreateData, collate_fn
from fragnet.model.gat.gat2 import FragNetFineTune

# Exercise the compiled scatter extension on CPU, without touching CUDA.
x = torch.tensor([1., 2.], device='cpu')
result = torch_scatter.scatter_add(x, torch.tensor([0, 0]), dim=0)
assert result.tolist() == [3.]
print(f'Environment verified: {sys.prefix}')
print(f'PyTorch {torch.__version__}; RDKit {rdkit.__version__}; PyG {torch_geometric.__version__}')
print('CPU dependency checks passed. Starting training.', flush=True)
PY

# --resume also works on the first submission when the output does not exist.
# Rejected graphs stop training for review; exclusions are not accepted silently.
exec srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    "$FRAGNET_PYTHON" -u "$TRAINING_SCRIPT" \
    --data-root "$PROJECT_DIR" \
    --fragnet-root "$PROJECT_DIR/smiles_baseline2/FragNet" \
    --device cpu \
    --threads "${SLURM_CPUS_PER_TASK:-8}" \
    --resume \
    --allow-graph-rejections

