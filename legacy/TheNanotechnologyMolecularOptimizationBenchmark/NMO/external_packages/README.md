# README
We provide some external packages for the user we have adapted to work with NMO.
Code is provided to ensure reproducibility of the results and to make it easier for users to use these packages. 
Note that the Licences defined for our Code does not apply and deviates from the ones of the external packages, so please check the licences of the external packages.

## Overview


| Package Name | Licence                    | Link                                     |
|--------------|----------------------------|------------------------------------------|
| dxtb         | Apache License             | [dxtb](https://github.com/grimme-lab/dxtb)       |
| xtb          | LGPL-3.0                   | [xtb](https://github.com/grimme-lab/xtb) |
| f-RAG        | NVIDIA Source Code License | [f-RAG](https://github.com/NVlabs/f-RAG) |
| mol_ga       | MIT License                | [mol_ga](https://github.com/AustinT/mol_ga) |
| gen_mol       | Apache License / NVIDIA Open Model License          | [gen_mol](https://github.com/NVIDIA-Digital-Bio/genmol/) |
|REINVENT4       | Apache License |[REINVENT4](https://github.com/MolecularAI/REINVENT4) |
|group-selfies       | Apache License |[group-selfies](https://github.com/aspuru-guzik-group/group-selfies) |
---

## Comments and Instructions for provided Baseline methods

### f-RAG
- We pretrained the Injection Module as described [here](https://github.com/NVlabs/f-RAG).
- A custom vocabulary was created using the `./f-RAG/get_vocab_custom.py` script provided.
- Code to run NMO oracles can be found in `./f-RAG/exps/nmo`
- The conda environment can be recreated using the `./f-RAG/environment.yaml` file. Tested with `Python 3.10.20`
- We used the seeds [0, 1, 2, 3, 4] for training. 

### mol_ga
- Check `./mol_GA/examples/USAGE.md` for usage instructions.

### GenMol 
- We pretrained the model according to the instructions for GenMol V1 [here](https://github.com/NVIDIA-Digital-Bio/genmol/) (Version 0.0.2).
- Initial vocabulary was created with `./genmol/scripts/exps/pmo_nmo/get_vocab_nmo.py`
- NMO runs can be started with `./genmol/scripts/exps/pmo_nmo/run_nmo.py`
- We used the seeds [0, 1, 2, 3, 4] for training.
- Tested with `Python 3.10.0`. The conda environment can be recreated using the `./genmol/environment.yaml` file. 


### REINVENT 
#### REINVENT (SMILES version)
* REINVENT is highly customizable and can be adapted to work with NMO oracles without changing the code by providing the proper plugin.
* NMO Plugin provided and all used configs file provided in `./REINVENT4/` (including priors)
* Tested with official REINVENT4 codebase (commit `5e67f40eedbb4c617710f8156b1db585cd789770`) and `Python 3.10.19`. The conda environment can be recreated using the `./REINVENT4/environment.yaml` file.
* We used the seeds [0, 1, 2, 3, 4] for training.
#### REINVENT GGS
* We adapted Reinvent to work with NMO and GGS encoding
* Adapted Files:
  * `reinvent/models/reinvent/models/vocabulary.py` 
  * `./REINVENT4_GGS/reinvent/runmodes/TL/reinvent.py`
  * `./REINVENT4_GGS/reinvent/runmodes/samplers/reinvent.py`
  * `./REINVENT4_GGS/reinvent/utils/config_parse.py`
  * `./REINVENT4_GGS/reinvent/chemistry/standardization/rdkit_standardizer.py`
  * NOTE: Reinvent checks validity of generated molecules with RDKit, which is not compatible with GGS encoding. We have added environment variables to disable these checks. Please set `export REINVENT_SKIP_FILTERS=1` and `export REINVENT_GRAMMAR_PATH` 
* Pretraining:  pretrained with `reinvent ./REINVENT4_GGS/reinvent_costom_dataset.toml`
* Training: Training with the provided *.toml REINVENT configuration files and *ini NMO config files
* We used the seeds [0, 1, 2, 3, 4] for training.