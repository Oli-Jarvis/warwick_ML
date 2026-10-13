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

If `run_manifest.json` is also present, the prediction script reads it for
additional target/source checks. `training_settings.json` is a training record;
it is not automatically read by that check.

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

## Generator files and analysis tools

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

## Run on Warwick

Large checkpoints, graph datasets and fitted reference bundles are kept on
Warwick. A code-only GitHub clone is not sufficient to run the workflow.
Use the existing `nmo` and `fragnet` Conda environments. Mapping requires the
RDKit, scikit-learn and DScribe versions recorded in the reference bundle.

```bash
cd /storage/msszkb_grp/msshfg
sbatch generate_and_map.sbatch
```

This CPU pilot checks five stored and regenerated graph predictions, then runs
with seed 123 and at most 256 oracle evaluations. Mapping runs in `fragnet` and
stops if its package versions differ from the reference. Each run name includes
the Slurm job ID to keep outputs separate.

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

## Scientific settings

The generator is a three-layer, 512-unit RNN. NMO retains genetic search, replay,
molecular validation, SA and rotatable-bond calculations. FragNet runs in a
persistent process using its own environment. Fitness uses predicted
`log_P_upconversion` with SA and rotatable-bond penalties: no extra logarithm
or molecular-length/area penalty. Failed predictions receive zero reward and
remain marked as failures. Per-run NMO source copies handle empty batches,
the evaluation budget and checkpoint saving without overwriting source files.

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
The launcher respects CPU/GPU selection, skips an unused grammar constructor
and runs every requested epoch. NMO supplies the training routine, GGS/descriptor
options, plots and logs. No manifest is created automatically; after validating
a new checkpoint, select it with generation's `--prior` option. Keep the supplied
prior and its matching manifest together.

Local checks cover syntax, shared-function equivalence, source patches and
synthetic mapping. Full inference, descriptor generation and Slurm execution
still require validation with the Warwick environments and saved data.
