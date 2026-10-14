#!/bin/bash
#SBATCH --job-name=fragnet-pilot
#SBATCH --partition=nanosimd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=fragnet-%j.log
#SBATCH --error=fragnet-%j.log

set -eo pipefail
source /software/easybuild/software/Miniconda3/4.12.0/etc/profile.d/conda.sh

ROOT=/storage/msszkb_grp/msshfg
SEED=123
CALLS=256
MODEL="$ROOT/fragnet_config_details"
REFERENCE="$ROOT/chemical_space_reference"
WORK="$ROOT/nmo_fragnet_generation"
FRAGNET_SOURCE="$ROOT/fragnet_source_code"
RUN="$WORK/runs/seed${SEED}_job${SLURM_JOB_ID}"
mkdir -p "$WORK/runs"
mkdir "$RUN"
mv "$SLURM_SUBMIT_DIR/fragnet-${SLURM_JOB_ID}.log" "$RUN/run.log"

for required in "$ROOT/generate_molecules.py" "$ROOT/map_chemical_space.py" \
                "$ROOT/initial_predictor_training/run_fragnet.py" \
                "$FRAGNET_SOURCE/fragnet/dataset/data.py" \
                "$MODEL/experiment/ft.pt" "$MODEL/graph_data/train.pkl" \
                "$MODEL/experiment/train_predictions.csv" "$MODEL/fragnet_selected.yaml" \
                "$REFERENCE/reference.pkl" "$REFERENCE/reference.html" \
                "$WORK/generator_prior/prior.pt" "$WORK/generator_prior/prior_manifest.json"; do
    if [[ ! -f "$required" ]]; then
        printf 'Missing input: %s\n' "$required" >&2
        exit 1
    fi
done

cd "$ROOT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=""

# Verify both models and the reference before generation.
conda activate fragnet
FRAGNET_PYTHON=$(python -c 'import sys; print(sys.executable)')
python -u generate_molecules.py verify \
    --run-dir "$MODEL" --fragnet-root "$FRAGNET_SOURCE" \
    --threads "$SLURM_CPUS_PER_TASK" --count 5
python -c 'import sys; from pathlib import Path; from map_chemical_space import load_reference; load_reference(Path(sys.argv[1]))' "$REFERENCE"

conda activate nmo
python -u generate_molecules.py \
    --run-dir "$MODEL" --fragnet-root "$FRAGNET_SOURCE" \
    --fragnet-python "$FRAGNET_PYTHON" \
    --repo "$ROOT/NMO_source_code" --work "$RUN" --name generation \
    --prior "$WORK/generator_prior/prior.pt" \
    --calls "$CALLS" --seed "$SEED" --threads "$SLURM_CPUS_PER_TASK" --cpu

conda activate fragnet
python -u map_chemical_space.py \
    --input "$RUN/generation/fragnet_predictions.csv" \
    --reference "$REFERENCE" --output "$RUN/space"

printf 'Completed. Plot: %s\n' "$RUN/space/candidates_in_space.html"
