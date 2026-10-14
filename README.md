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

Entry points reside at the project root. Python defaults resolve relative to it;
set the project path in the Slurm file and pretraining INI for your installation.

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

## FragNet model and performance

The current predictor learns xTB-derived `log_P_upconversion` from the combined
SMILES/GGS data. The selected configuration is recorded in
`fragnet_config_details/fragnet_selected.yaml` and `training_settings.json`.

| Hyperparameter | Selected value |
| --- | --- |
| Architecture | FragNet `gat2`, 4 layers, 4 attention heads, embedding dimension 128 |
| Input features | 167 atom/fragment features; 17 bond features |
| Regression head | `FTHead3`; configured widths 64, 1920, 1472, 2048; ReLU |
| Dropout | 0.25 |
| Optimiser | Adam |
| Learning rate | 0.0007905475075930115 |
| Batch size | 128 |
| Objective | Mean squared error on `log_P_upconversion` |
| Training limit | 10,000 epochs; early-stopping patience 100 |
| Learning-rate scheduler | Disabled |
| Seed | 123 |
| Checkpoint selection | Lowest validation MSE; test data excluded from selection |

These are the property-training settings; the YAML's separate `pretrain` block
describes the FragNet backbone configuration. Neither is the NMO generator prior.

Molecules are grouped by stereo-free identity when allocating splits.
Targets are the mean of distinct observed values per canonical isomeric SMILES.
Graph construction retains specified tetrahedral and double-bond stereochemistry.

### Performance

Final train, validation and test metrics are saved in
`fragnet_config_details/experiment/regression_metrics.json`, with per-molecule
predictions alongside it. Numerical results are pending inclusion from the SCRTP
results file. Report MAE, RMSE and R² for each split, together with the selected
epoch and evaluated sample counts; errors refer to the logarithmic target.

The training curve records optimisation progress, not held-out test performance.
The similarity analysis in `fragnet_analysis/tanimoto_similarity/` helps assess
how prediction error changes with similarity to training molecules. Agreement
with xTB targets does not establish accuracy against DFT.

## Next steps: DFT fine-tuning

The intended next stage is to fine-tune FragNet on DFT-derived targets. The xTB
model provides a baseline and an initial checkpoint for this transfer. Since xTB
is already inexpensive in this workflow, the main scientific motivation is to
approximate the more costly DFT target; any practical speed advantage over direct
xTB evaluation should be measured.

1. Assemble a chemically diverse DFT dataset, including promising generated
   candidates. Record the calculation protocol, target definition and units.
2. Fine-tune the xTB-trained FragNet checkpoint using DFT labels. Select the
   learning rate, freezing strategy and stopping point on validation data;
   reserve an independent molecularly grouped DFT test set.
3. Compare held-out DFT errors and candidate rankings against the xTB baseline,
   and validate selected generated molecules with explicit DFT calculations.
4. Integrate the validated predictor into generation, retaining the existing NMO
   prior where vocabulary and model compatibility are unchanged. Changing the
   predictor alone does not require generator pretraining.

The current reference uses an xTB-derived property in both clustering and the
display's third axis. DFT predictions must be labelled distinctly and should not
be passed into that clustering coordinate as though they were xTB values.
Before DFT-based mapping, choose a consistent property basis for reference and
candidates, or construct a separate reference. Structural PCA transforms can be
reused where descriptor definitions and preprocessing remain unchanged.

DFT fine-tuning and the corresponding mapping changes are planned work; the
supplied workflow currently uses the xTB-trained predictor.

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

Analysis tools are independent of the generation/mapping pipeline and accept
explicit input/output paths.

## Dependencies and reproducibility

| Environment | Tasks | Principal dependencies |
| --- | --- | --- |
| `nmo` | Generation and generator pretraining. | NMO dependencies, including PyTorch, RDKit, NumPy, pandas, Matplotlib, tqdm, GGS and group-selfies. |
| `fragnet` | Predictor training, verification, prediction and mapping. | PyTorch, PyTorch Geometric, torch-scatter, RDKit, NumPy, pandas, PyYAML, NetworkX, SciPy, scikit-learn and LMDB. Mapping also requires ASE, DScribe and Plotly. |

Generation launches a persistent prediction worker using the `fragnet` Python
executable while NMO runs in `nmo`. The Slurm script handles environment selection.
Mapping enforces the RDKit, scikit-learn and DScribe versions saved in the reference.

### Environment specifications

Export the working SCRTP installations and commit the reviewed YAML files:

```bash
conda env export -n nmo | sed '/^prefix:/d' > environment-nmo.yml
conda env export -n fragnet | sed '/^prefix:/d' > environment-fragnet.yml
```

Retain package versions, channels and pip dependencies. Replace account-specific
local/editable-install paths with the appropriate source locations and test
recreation before distributing the specifications.

Recreate from the project checkout:

```bash
conda env create -f environment-nmo.yml
conda env create -f environment-fragnet.yml
```

The YAML files still need exporting and validation on SCRTP. Checkpoints,
graph datasets and reference bundles are distributed separately from the environments.

## Run the workflow

Required checkpoints, graph datasets and reference bundles are stored on SCRTP
and are separate from the GitHub code checkout.

```bash
cd /storage/msszkb_grp/msshfg
sbatch generate_and_map.sbatch
```

This CPU pilot checks five stored and regenerated graph predictions, then runs
with seed 123 and at most 256 oracle evaluations. Mapping runs in `fragnet` and
stops if its package versions differ from the reference. Each run name includes
the job ID.

### Run seed

`--seed 123` is an arbitrary default; valid generation seeds are integers from
0 to 2**32 - 1. Use different seeds for independent search replicates, retaining
the same prior and predictor checkpoint. Update `RUN_NAME` in the batch script
to match the chosen seed. The predictor's graph-construction seed is taken
separately from `fragnet_selected.yaml`. Reproducibility also depends on the
software, hardware and execution configuration.

| Output under `nmo_fragnet_generation/` | Contents |
| --- | --- |
| `pilot_seed123_JOBID/` | `full_history.csv`, adapted generator `agent_final.pt`, `fragnet_predictions.csv`, `unique_fragnet_candidates.csv`, configuration, logs and audit summaries. |
| `pilot_seed123_JOBID_space/` | `candidates_in_space.html`, `candidates_in_space.csv`, generated conformers, rejected rows and projection summary. |

Independent scoring and projection, using the `fragnet` environment and a CSV
with a `smiles` column:

```bash
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

For deliberate retraining, using the `nmo` environment:

```bash
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
still require validation with the SCRTP environments and saved data.
