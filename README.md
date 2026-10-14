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

FragNet predicts xTB-derived `log_P_upconversion` from combined SMILES/GGS data.
Full settings are in `fragnet_config_details/fragnet_selected.yaml` and
`training_settings.json`.

| Setting | Selected value |
| --- | --- |
| Architecture | `gat2`; 4 layers, 4 attention heads, embedding dimension 128 |
| Features | 167 atom/fragment features; 17 bond features |
| Regression head | `FTHead3`; configured widths 64, 1920, 1472, 2048; ReLU |
| Optimisation | Adam; learning rate ≈ 7.9055 × 10⁻⁴; batch size 128; MSE loss |
| Regularisation | Dropout 0.25; no learning-rate scheduler |
| Training | Seed 123; maximum 10,000 epochs; early-stopping patience 100 |

These are the property-training settings, separate from FragNet backbone
pretraining and the NMO generator prior. Splits group stereo-free molecular
identities; targets average distinct observed values per canonical isomeric SMILES.
Graphs retain specified tetrahedral and double-bond stereochemistry.

The checkpoint at **epoch 100** had the lowest validation MSE. Test data were
not used for selection. Results from `experiment/regression_metrics.json`:

| Split | Molecules | MAE | RMSE | R² | Spearman ρ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 31,766 | 1.104 | 1.590 | 0.934 | 0.961 |
| Validation | 3,970 | 1.329 | 2.032 | 0.805 | 0.916 |
| Test | 3,960 | 1.262 | 1.925 | 0.806 | 0.915 |

Errors are on the logarithmic target scale. Similar validation/test results and
test rank correlation support prediction within the evaluated dataset; accuracy
on generated molecules or against DFT requires separate validation.
Kernel ridge regression (KRR) was also trialled as the predictor but performed
less well in the experiments conducted, so FragNet was retained.

## Next steps: DFT fine-tuning

Since xTB is already fast, the main next objective is a surrogate for DFT-derived
properties. The xTB-trained FragNet provides an initial checkpoint and baseline;
any speed advantage over direct xTB evaluation should be measured.

- Assemble diverse DFT-labelled molecules with a consistent calculation protocol,
  target definition and units.
- Fine-tune FragNet, selecting learning rate, frozen layers and stopping point
  on validation data; evaluate errors and rankings on a separate molecularly
  grouped DFT test set and check selected generated candidates explicitly with DFT.
- Integrate the validated predictor. The existing NMO prior can be reused if
  vocabulary and model compatibility are unchanged.

The reference clusters and third plot axis include an xTB-derived property.
DFT-based mapping therefore needs a consistent property basis for reference and
candidates, or a separate reference; DFT values cannot simply replace xTB values.
Saved descriptor/PCA transforms remain reusable with unchanged preprocessing.
These are planned steps; the current workflow uses the xTB-trained model.

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

Distribute reviewed, tested YAML exports with the code. Checkpoints, graph
datasets and reference bundles are separate from the environments.

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
