#!/bin/bash
#SBATCH --job-name=fragnet-pilot
#SBATCH --partition=nanosimd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=fragnet-pilot-%j.log
#SBATCH --error=fragnet-pilot-%j.err

set -eo pipefail

source /software/easybuild/software/Miniconda3/4.12.0/etc/profile.d/conda.sh

ROOT=/storage/msszkb_grp/msshfg
SCRIPTS="$ROOT"
MODEL="$ROOT/fragnet_config_details"
REFERENCE="$ROOT/chemical_space_reference"
WORK="$ROOT/nmo_fragnet_generation"
FRAGNET_SOURCE="$ROOT/fragnet_source_code"
RUN_NAME="pilot_seed123_${SLURM_JOB_ID}"

# If relocated, set FRAGNET_SOURCE to the directory containing the fragnet package.
if [[ ! -f "$FRAGNET_SOURCE/fragnet/dataset/data.py" ]]; then
    printf 'FragNet source not found at %s; set FRAGNET_SOURCE in this script.\n' "$FRAGNET_SOURCE" >&2
    exit 1
fi
for required in "$SCRIPTS/generate_molecules.py" "$SCRIPTS/map_chemical_space.py" \
                "$MODEL/experiment/ft.pt" "$MODEL/graph_data/train.pkl" \
                "$MODEL/experiment/train_predictions.csv" "$MODEL/fragnet_selected.yaml" \
                "$REFERENCE/reference.pkl" "$REFERENCE/reference.html" \
                "$WORK/generator_prior/prior.pt" "$WORK/generator_prior/prior_manifest.json"; do
    if [[ ! -f "$required" ]]; then
        printf 'Missing input: %s\n' "$required" >&2
        exit 1
    fi
done

cd "$SCRIPTS"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=""

conda activate fragnet
FRAGNET_PYTHON=$(python -c 'import sys; print(sys.executable)')
python -u generate_molecules.py verify \
    --run-dir "$MODEL" --fragnet-root "$FRAGNET_SOURCE" \
    --threads "$SLURM_CPUS_PER_TASK" --count 5

conda activate nmo
python -u generate_molecules.py \
    --run-dir "$MODEL" --fragnet-root "$FRAGNET_SOURCE" \
    --fragnet-python "$FRAGNET_PYTHON" \
    --repo "$ROOT/NMO_source_code" --work "$WORK" --name "$RUN_NAME" \
    --calls 256 --seed 123 --threads "$SLURM_CPUS_PER_TASK" --cpu

conda activate fragnet
python -u map_chemical_space.py \
    --input "$WORK/$RUN_NAME/unique_fragnet_candidates.csv" \
    --reference "$REFERENCE" \
    --output "$WORK/${RUN_NAME}_space"

printf 'Completed. Plot: %s\n' "$WORK/${RUN_NAME}_space/candidates_in_space.html"
