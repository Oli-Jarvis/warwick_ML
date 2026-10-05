#create conda env
conda create -n nmo python=3.11

conda activate nmo

#install xtb
conda install -y -c conda-forge xtb==6.7.1

#install mkl runtime for xtb binary
conda install -y -c https://software.repos.intel.com/python/conda/ -c conda-forge mkl

#download electrode data
pip install huggingface_hub
hf download guise868/electrode_data --repo-type=dataset --local-dir ./NMO/data/

#install dxtb
pip install ./NMO/external_packages/dxtb

#install group-selfies
pip install ./NMO/external_packages/group-selfies

#install GGS
pip install ./GGS/

#install dxtb (NMO dependency, must be installed before NMO)
pip install ./NMO/external_packages/dxtb/

#install NMO
pip install ./NMO/

#install requirements of the baseline method
pip install -r ./genetic_GFN_framework/requirements.txt


#setup xtb for upconversion calculation
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export XTB_PTB_BIN=$(pwd)/NMO/external_packages/xtb/xtb