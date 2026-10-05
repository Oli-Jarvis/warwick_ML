#!/bin/bash
set -euo pipefail

##############################################################################
# NMO Environment Setup & xtb Build Script
# Sets up conda env, installs dependencies, builds xtb from source,
# and configures the NMO benchmark project.
##############################################################################

PROJECT_ROOT="$(pwd)"
ENV_NAME="nmo"
IMKL_LIB="/software/easybuild/software/imkl/2025.1.0/mkl/2025.1/lib"

echo ">>> [1/7] Loading HPC modules"
module purge
module load GCCcore/12.3.0 Python/3.11.3
module load CMake
module load imkl

export PATH=/software/easybuild/software/Miniconda3/4.12.0/bin:$PATH

echo ">>> [2/7] Configuring conda directories"
mkdir -p ~/.conda/pkgs ~/.conda/envs
cat > ~/.condarc << 'EOF'
pkgs_dirs:
  - ~/.conda/pkgs
envs_dirs:
  - ~/.conda/envs
EOF

echo ">>> [3/7] Creating and activating conda environment: $ENV_NAME"
if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -y -n "$ENV_NAME" python=3.11
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo ">>> [4/7] Installing core dependencies via conda-forge"
conda install -y -c conda-forge mkl
conda install -y -c conda-forge xtb==6.7.1
conda install -y -c conda-forge toml-f mctc-lib

pip install huggingface_hub

echo ">>> [5/7] Downloading dataset"
hf download guise868/electrode_data \
    --repo-type=dataset \
    --local-dir "$PROJECT_ROOT/NMO/data/"

echo ">>> [6/7] Installing local project packages"
pip install "$PROJECT_ROOT/NMO/external_packages/dxtb"
pip install "$PROJECT_ROOT/NMO/external_packages/group-selfies"
pip install "$PROJECT_ROOT/GGS/"
pip install "$PROJECT_ROOT/NMO/"
pip install -r "$PROJECT_ROOT/genetic_GFN_framework/requirements.txt"

echo ">>> [7/7] Building xtb from source"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd "$PROJECT_ROOT/NMO/external_packages/xtb"

# Compatibility shim: old 'tomlf' name -> installed 'toml-f' config
if [ ! -f "$CONDA_PREFIX/lib/cmake/tomlf/tomlf-config.cmake" ]; then
    mkdir -p "$CONDA_PREFIX/lib/cmake/tomlf"
    cat > "$CONDA_PREFIX/lib/cmake/tomlf/tomlf-config.cmake" << 'EOF'
include(${CMAKE_CURRENT_LIST_DIR}/../toml-f/toml-f-config.cmake)
EOF
fi

[ -d xtb ] && rm -rf xtb
[ -d build ] && rm -rf build

mkdir -p build

cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$IMKL_LIB" \
    -DCMAKE_INSTALL_RPATH="$IMKL_LIB"

make -C build

# make -C build test || echo "WARNING: xtb test suite reported failures, review before proceeding."

cp build/xtb .

cd "$PROJECT_ROOT"

export XTB_PTB_BIN="$PROJECT_ROOT/NMO/external_packages/xtb/xtb"

echo ""
echo "=== Setup complete ==="
echo "Conda env:      $ENV_NAME"
echo "xtb binary:     $XTB_PTB_BIN"
echo ""
echo "To run:"
echo "conda activate nmo"
echo "cd NMO/example/SMILES_minimal_example"
blaschma
/
TheNanotechnologyMolecularOptimizationBenchmarkecho "python smiles_minimal_example_upconversion.py config_upconversion.ini"
