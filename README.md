# Molecular generation with FragNet

This project generates molecules with NMO, scores them using a trained FragNet
property predictor, and maps them onto a fixed SMILES/GGS chemical space.

## Workflow

1. **Prepare the models and reference once:** train FragNet, pretrain the NMO
   generator, and build the chemical-space reference.
2. **Generate and predict:** start from the saved generator prior. FragNet predicts
   `log_P_upconversion` during the search; these predictions guide molecular fitness.
3. **Map the results:** calculate candidate descriptors, apply the saved PCA
   transforms and assign candidates to the reference clusters.

The supplied models and reference are already prepared. A normal generation run
reuses them. Generation and mapping are separate steps; mapping does not guide
the search towards a particular cluster.

## Main files

| File | Purpose |
| --- | --- |
| `generate_molecules.py` | Runs generation with FragNet fitness. Also supports independent `predict` and `verify` commands. Imports graph/model functions from `initial_predictor_training/run_fragnet.py` without starting predictor training. |
| `map_chemical_space.py` | Maps scored candidates onto the saved reference and produces a plot and CSV. |
| `pretrain_generator.py` | Runs NMO generator pretraining. Needed only when deliberately creating another prior. |
| `generate_and_map.sbatch` | Slurm job: verify predictions, generate molecules, then map them. |
| `nmo_fragnet_generation/pretrain.ini` | Generator-pretraining settings and output location. |

Keep these Python entry points and the Slurm file at the project root, alongside
the directories below. Python defaults follow this location. The Slurm file and
pretraining INI use `/storage/msszkb_grp/msshfg`; set this path for your installation.

## Source code and predictor training

| Path | Purpose |
| --- | --- |
| `NMO_source_code/` | Molecular generator, genetic search and molecular-processing implementations. |
| `NMO_source_code/genetic_GFN_framework/train.py` | Generator optimisation and genetic-search loop. |
| `NMO_source_code/genetic_GFN_framework/pretrain.py` | Learns a generator prior from molecular sequences. |
| `NMO_source_code/genetic_GFN_framework/data/experiments/Voc_adapted` | Generator vocabulary; token ordering must match the saved prior. |
| `NMO_source_code/genetic_GFN_framework/data/experiments/translated_smiles.smi` | Molecular sequences used for generator pretraining. |
| `fragnet_source_code/` | FragNet architecture, molecular graph construction and feature definitions. |
| `initial_predictor_training/run_fragnet.py` | Prepares data, builds graphs, trains/evaluates FragNet and provides shared graph/model functions. |
| `initial_predictor_training/run_fragnet.sh` | Slurm submission script for predictor training. Set its script, source and data paths before submitting. |

### `fragnet_config_details/`

| File or directory | Purpose |
| --- | --- |
| `fragnet_selected.yaml` | Predictor architecture and training configuration. Inference loads the checkpoint from the supplied model directory. |
| `training_settings.json` | Training settings, input/source hashes, target definition, split policy and stereochemical policy. |
| `splits/` | Molecular datasets assigned to training, validation and testing. |
| `graph_data/` | Prepared graph datasets, including `train.pkl`, `val.pkl` and `test.pkl`. Verification uses `train.pkl`. |
| `graph_rejected.csv` | Molecules that could not be converted into accepted graphs, with rejection details. |
| `experiment/ft.pt` | Best-validation predictor checkpoint used for inference. |
| `experiment/last.pt` | Training checkpoint for resuming predictor training. |
| `experiment/training_history.csv` | Training, validation and best-validation MSE by epoch. |
| `experiment/regression_metrics.json` | Predictor evaluation metrics. |
| `experiment/train_predictions.csv`, `val_predictions.csv`, `test_predictions.csv` | Predictions and targets for evaluating each split. Verification uses training predictions. |

## Chemical-space reference

All paths in this table are beneath `chemical_space_reference/`.

| File or directory | Purpose |
| --- | --- |
| `smiles_runs/`, `ggs_runs/` | Molecular run data used to construct the reference. |
| `dim.py` | Calculates bonding and structural descriptors, fits PCA and saves the fitted preprocessing/transforms. |
| `dim_reduction_details/` | Descriptor caches, PCA embeddings and fitted PCA data used to construct the reference. |
| `analyse.py` | Clusters the PCA/property representation using BIRCH followed by agglomerative clustering, and exports the reference. |
| `cluster_plots/` | Reference-cluster plots and analysis tables. |
| `clustering_feature_scaling.csv` | Centering and scaling applied to each clustering feature. |
| `reference.pkl` | Saved preprocessing, PCA, clustering models and reference data used for candidate mapping. |
| `reference.csv` | Reference molecules, coordinates, properties and cluster labels. |
| `reference.html` | Interactive reference plot used as the background for candidate overlays. |
| `reference.json` | Reference counts, settings and provenance. |

The reference contains 68,268 rows: 36,754 SMILES and 31,514 GGS, with 20 final
clusters. Assignment uses three bonding PCs, three structural PCs and log P,
with the saved scaling. The plot displays bonding PC1, structural PC1 and log P.
Mapping applies saved transforms and labels; it does not refit PCA or recluster.

## Saved generator and analysis tools

A **prior** is the pretrained generator used as the starting point for a search.
`prior.pt` contains its learned parameters. Each generation run loads this file
and saves its adapted generator separately as `agent_final.pt`; the starting
prior is retained. A new search or a different random seed does not require
pretraining again.

| Path | Purpose |
| --- | --- |
| `nmo_fragnet_generation/generator_prior/prior.pt` | Pretrained RNN parameters: the starting point for a generation run. |
| `nmo_fragnet_generation/generator_prior/prior_manifest.json` | Prior settings and checkpoint/vocabulary hashes used to check compatibility. |
| `generator_prior/valid_ratio.txt`, `valid_ratio.pdf` | Pretraining diagnostics showing sampled-molecule validity; inside `nmo_fragnet_generation/`. |
| `generator_prior/n_atoms.txt`, `n_atoms_histogram.pdf` | Atom counts and their distribution for valid sampled molecules; inside `nmo_fragnet_generation/`. |
| `fragnet_analysis/embeddings.py` | Creates an offline HTML report of FragNet fragment and fragment-connection contributions. |
| `fragnet_analysis/neighbours.py` | Finds similar molecules using embedding cosine similarity or Morgan-fingerprint Tanimoto similarity, depending on the input. |
| `fragnet_analysis/tanimoto_similarity/` | Tables and plots relating predictor error to molecular similarity. |

Optional analysis tools are separate from generation and mapping. Supply their
input/output paths explicitly; inspect `--help` before running them.

## Python environments

A Conda environment is a separate Python installation with its own packages.
`conda activate NAME` selects that environment's Python and libraries. The names
`nmo` and `fragnet` identify the two environments used by the batch script; they
are not model files and are not created by cloning this repository.

| Environment | When to use it | Key software required by the code |
| --- | --- | --- |
| `nmo` | Generation and optional generator pretraining. | The group's NMO dependencies, including PyTorch, RDKit, NumPy, pandas, Matplotlib, tqdm, GGS and group-selfies. The entry points locate NMO/GGS code in `NMO_source_code/`. |
| `fragnet` | FragNet training, verification, standalone prediction, the prediction worker, and mapping in the supplied batch job. | PyTorch, PyTorch Geometric, torch-scatter, RDKit, NumPy, pandas, PyYAML, NetworkX, SciPy, scikit-learn, LMDB and the code in `fragnet_source_code/`. Mapping additionally uses ASE, DScribe and Plotly. |

These are the main dependencies, not an export of every installed package.
To inspect the actual installations and versions on Warwick:

```bash
conda list -n nmo
conda list -n fragnet
```

During generation, NMO runs in `nmo` and the script starts a persistent FragNet
worker using the Python executable from `fragnet`. This keeps their dependencies
separate. The Slurm script activates the environments automatically; for manual
prediction/mapping use `conda activate fragnet`, and for pretraining use
`conda activate nmo`. Optional analysis tools need the packages used by that tool.

Mapping requires the RDKit, scikit-learn and DScribe versions recorded in the
saved reference bundle. Reproduce the working environments for another machine
rather than assuming that arbitrary package versions will be compatible.

### Share and recreate the working environments

For group members using Warwick's Linux system, share exports of the tested
`nmo` and `fragnet` environments. Each member creates their own copies.
The exports record package versions, including pip packages where detected;
they do not contain checkpoints, data or the project source directories.

**Maintainer: export once from the working Warwick installation.**

```bash
cd /storage/msszkb_grp/msshfg
conda env export -n nmo | sed '/^prefix:/d' > environment-nmo.yml
conda env export -n fragnet | sed '/^prefix:/d' > environment-fragnet.yml
```

Review both YAML files before adding them to GitHub. Removing `prefix:` avoids
using the maintainer's environment location. Also replace any account-specific
paths, local `file://` entries or editable-install paths with the member's source
location and document any required local installation. Preserve package versions,
channels and necessary pip packages. Test the exports in a separate installation
before calling them ready for the group; the YAML files still need to be exported
from Warwick and checked.

**Group member: create the environments from the shared YAML files.**

From the project checkout, with Conda available:

```bash
source /software/easybuild/software/Miniconda3/4.12.0/etc/profile.d/conda.sh
conda env create -f environment-nmo.yml
conda env create -f environment-fragnet.yml
```

These commands expect the names `nmo` and `fragnet` to be available in your account.
The full exports target the working Warwick Linux setup; other operating systems
or architectures may need adjusted package builds. Use the project's source code
and shared data paths alongside these environments.

Before a generation job, verify the recreated predictor environment:

```bash
conda activate fragnet
python generate_molecules.py verify --run-dir /storage/msszkb_grp/msshfg/fragnet_config_details --fragnet-root ./fragnet_source_code --count 5
```

This requires access to the saved checkpoint, training graphs and training
predictions. The mapping step also checks its package versions against the
reference bundle. The batch job performs prediction verification again before
starting generation. It activates both environments as needed; manual commands
use the activation instructions above.

Conda reference: [Exporting and creating environments](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html).

## Run on Warwick

Large checkpoints, graph datasets and fitted reference bundles are kept on
Warwick. A code-only GitHub clone is not sufficient to run the workflow.

```bash
cd /storage/msszkb_grp/msshfg
sbatch generate_and_map.sbatch
```

This CPU pilot checks five stored and regenerated graph predictions, then runs
with seed 123 and at most 256 oracle evaluations. Mapping runs in `fragnet` and
stops if its package versions differ from the reference. Each run name includes
the Slurm job ID to keep outputs separate.

### Choosing a seed

A random seed sets the starting state of the random-number generators. **123 is
an arbitrary default**, not a preferred molecule, cluster or scientifically
optimised value. Other integers from 0 to 2**32 - 1 can be used, such as 0, 1 or 42.
Keeping the seed and setup fixed helps reproduce a run; changing the seed samples
a different search. Exact reproducibility can also depend on software, hardware
and parallel execution.

To change the generation seed, edit `--seed 123` in `generate_and_map.sbatch` and
update its `RUN_NAME` label to match. For example, use `--seed 42` with
`RUN_NAME="pilot_seed42_${SLURM_JOB_ID}"`. This reuses the same prior and FragNet
checkpoint. The predictor's graph-construction seed is read separately from
`fragnet_selected.yaml`.

| Output under `nmo_fragnet_generation/` | Contents |
| --- | --- |
| `pilot_seed123_JOBID/` | `full_history.csv`, adapted generator `agent_final.pt`, `fragnet_predictions.csv`, `unique_fragnet_candidates.csv`, configuration, logs and audit summaries. |
| `pilot_seed123_JOBID_space/` | `candidates_in_space.html`, `candidates_in_space.csv`, generated conformers, rejected rows and projection summary. |

To score a separate CSV containing a `smiles` column, choose new output directories:

```bash
conda activate fragnet
python generate_molecules.py predict --input molecules.csv --output predictions_new
python map_chemical_space.py --input predictions_new/predictions.csv --output mapped_new
```

## FragNet integration

The workflow calls NMO's search implementation and replaces the property
calculation with FragNet predictions. It does not replace NMO's search algorithm;
per-run source copies handle empty batches, the evaluation budget and checkpoint
saving without overwriting the NMO sources.

FragNet loads `experiment/ft.pt` using the architecture in `fragnet_selected.yaml`.
Molecular graph construction and stereochemistry checks use the same functions
as predictor training. Its output is `log_P_upconversion`, used directly without
another logarithm. The configured fitness applies SA and rotatable-bond penalties,
with no molecular-length/area penalty. Failed predictions receive zero reward
and remain marked as failures. Prediction verification compares saved and newly
constructed graphs with stored training predictions before the batch starts.

Predicted properties are not xTB measurements. Mapping and cluster assignment
alone do not establish molecular novelty or prediction accuracy.

## Optional generator pretraining

Normal runs do not require this step:

```bash
conda activate nmo
python pretrain_generator.py nmo_fragnet_generation/pretrain.ini --seed 123
```

The INI selects six epochs, batch size 128, learning rate 0.0005 and disabled
masking. Output goes to `generator_prior_new`; a nonempty destination is refused.
This calls the NMO pretraining routine and retains its training options and
outputs. No manifest is created automatically; after validating
a new checkpoint, select it with generation's `--prior` option. Keep the supplied
prior and its matching manifest together.

Local checks cover syntax, shared-function equivalence, source patches and
synthetic mapping. Full inference, descriptor generation and Slurm execution
still require validation with the Warwick environments and saved data.
