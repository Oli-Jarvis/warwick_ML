# Molecular generation with FragNet

This project generates molecules with NMO, scores them using a trained FragNet
property predictor, and maps them onto a fixed SMILES/GGS chemical space.

## Workflow

1. **Prepare the models and reference once:** train FragNet, pretrain the NMO
   generator, and build the chemical-space reference.
2. **Generate and predict:** start from the saved generator prior. FragNet predicts
   `log_P_upconversion` during the search; these predictions enter the fitness calculation used to select molecules.
3. **Map the results:** calculate candidate descriptors, apply the saved PCA
   transforms and assign candidates to the reference clusters.

The supplied models and reference are already prepared. A normal generation run
reuses them. Generation and mapping are separate steps; mapping does not guide
the search towards a particular cluster.

## Scientific workflow

### How is the generator pretrained?

The supplied **prior** is a three-layer RNN with hidden size 512, pretrained on
the molecular sequences in `translated_smiles.smi` using the matching
`Voc_adapted` vocabulary. It learns to predict successive tokens in these
sequences by minimising their mean negative log-likelihood. This learns the
sequence distribution of the training molecules; no upconversion labels or
FragNet predictions enter this pretraining loss.

The supplied setup uses six epochs, batch size 128, initial Adam learning rate
0.0005 and maximum sequence length 140, with masking and descriptor prediction
disabled. An **epoch** is one complete pass through the training dataset.
The saved parameters are in `generator_prior/prior.pt` under
`nmo_fragnet_generation/`. Each search starts from this prior and saves its
updated generator separately as `agent_final.pt`. Changing the search seed or
replacing the property predictor does not itself require a new prior.

### What is being optimised?

FragNet is trained to minimise mean squared error against xTB-derived
`log_P_upconversion`. During generation its weights remain fixed: NMO updates
the generator and performs genetic search using FragNet predictions as part of
the molecular fitness.

For a successfully scored molecule, let $y$ be the predicted
`log_P_upconversion`, $s$ the synthetic-accessibility score (SA), and $r$
the number of rotatable bonds. The implemented fitness is

$$
F = \frac{1}{4}\log_{10}(1+10^c)
    \left(\frac{10-s}{9}\right)^2
    \frac{1}{1+\exp[2(r-3.5)]},
\qquad c=\operatorname{clip}(y+5,-50,80).
$$

The search favours higher predicted upconversion, lower SA scores and fewer
rotatable bonds. The logarithmic term gives a smooth positive reward; clipping
bounds extreme predictions. There is no molecular-length or surface-area
penalty. Failed predictions receive zero fitness and remain marked as failures.

NMO retains its molecular validation and anchoring rules. FragNet predicts the
logarithmic property directly; it is passed to the reward transformation above
without first taking another logarithm. Its graph
construction and stereochemistry handling match predictor training. Candidate
geometries used for graphs and descriptors are generated with RDKit; this run
does not evaluate candidate upconversion with xTB or DFT.

The resulting fitness is a search objective, not a measured physical property.
High-scoring candidates still need independent property calculations and checks
against the training/reference molecules to establish accuracy and novelty.
Cluster membership is used for analysis after generation, not as a search reward.

## Main files

| File | Purpose |
| --- | --- |
| `generate_molecules.py` | Runs generation using FragNet predictions in the fitness calculation. Also supports independent `predict` and `verify` commands. |
| `map_chemical_space.py` | Maps scored candidates onto the saved reference and produces a plot and CSV. |
| `pretrain_generator.py` | Runs NMO generator pretraining. Use only to train a new starting generator. |
| `generate_and_map.sbatch` | Slurm job: verify predictions, generate molecules, then map them. |
| `nmo_fragnet_generation/pretrain.ini` | Generator-pretraining settings and output location. |

Keep the main scripts in the top-level project directory, alongside
`NMO_source_code/` and `fragnet_config_details/`. Python scripts find default
inputs relative to their own location. If you move the project, update the paths
in `generate_and_map.sbatch` and `nmo_fragnet_generation/pretrain.ini`.

## Saved predictor files

All paths in this table are relative to `fragnet_config_details/`.

| File or directory | Purpose |
| --- | --- |
| `fragnet_selected.yaml` | FragNet architecture and training settings. Prediction uses `experiment/ft.pt` in this directory. |
| `training_settings.json` | Training settings, hashes identifying input files and source code, target definition, split rules and stereochemistry handling. |
| `splits/` | Molecular datasets assigned to training, validation and testing. |
| `graph_data/` | Molecular graphs saved as `train.pkl`, `val.pkl` and `test.pkl`. The prediction check uses `train.pkl`. |
| `graph_rejected.csv` | Molecules rejected during graph construction, with reasons. |
| `experiment/ft.pt` | Saved FragNet parameters from the epoch with lowest validation MSE; used for prediction. |
| `experiment/last.pt` | Training checkpoint for resuming predictor training. |
| `experiment/training_history.csv` | Training MSE, validation MSE and lowest validation MSE so far, for each epoch. |
| `experiment/regression_metrics.json` | Predictor evaluation metrics. |
| `experiment/train_predictions.csv`, `val_predictions.csv`, `test_predictions.csv` | Predictions and targets for evaluating each split. Verification uses training predictions. |

## FragNet model and performance

FragNet predicts xTB-derived `log_P_upconversion` from combined SMILES/GGS data.
Full settings are in `fragnet_config_details/fragnet_selected.yaml` and
`fragnet_config_details/training_settings.json`.

| Setting | Selected value |
| --- | --- |
| Architecture | `gat2`; 4 layers, 4 attention heads, embedding dimension 128 |
| Features | 167 atom/fragment features; 17 bond features |
| Regression head | `FTHead3`; configured widths 64, 1920, 1472, 2048; ReLU |
| Optimisation | Adam; learning rate ≈ 7.9055 × 10⁻⁴; batch size 128; MSE loss |
| Regularisation | Dropout 0.25; no learning-rate scheduler |
| Training | Seed 123; maximum 10,000 epochs; stop after 100 consecutive epochs without validation improvement |

These are the property-training settings, separate from FragNet backbone
pretraining and the NMO generator prior. Molecules differing only in stereochemistry stay in the same dataset split to
prevent overlap between training, validation and test sets. Each canonical
isomeric SMILES receives the mean of its distinct observed target values.
Graphs retain specified tetrahedral and double-bond stereochemistry.

The checkpoint at **epoch 100** had the lowest validation MSE. Test data were
not used for selection. Results from `fragnet_config_details/experiment/regression_metrics.json`:

| Split | Molecules | MAE | RMSE | R² | Spearman ρ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 31,766 | 1.104 | 1.590 | 0.934 | 0.961 |
| Validation | 3,970 | 1.329 | 2.032 | 0.805 | 0.916 |
| Test | 3,960 | 1.262 | 1.925 | 0.806 | 0.915 |

MAE and RMSE measure errors in `log_P_upconversion`; Spearman ρ measures agreement
in molecular rankings. Validation and test errors are similar. These results do
not establish accuracy for newly generated molecules or DFT-derived targets.
Kernel ridge regression (KRR) was also trialled as the predictor but performed
less well in the experiments conducted, so FragNet was retained.

## Next steps: DFT fine-tuning

We already have approximately 200 molecules with DFT data for fine-tuning.
Since xTB is already fast, the next objective is to predict DFT-derived properties
without running DFT for every candidate. Fine-tuning will start from the
xTB-trained FragNet parameters, with the current model used for comparison;
any speed advantage over direct xTB evaluation should be measured.

- Prepare the existing DFT dataset for fine-tuning, checking that calculation protocols, target definitions and units
  are consistent. With this small dataset, use grouped cross-validation within
  the training data for model selection and retain an independent DFT test set.
- Fine-tune FragNet, using validation data to select the learning rate, which layers
  to leave unchanged and when to stop training; evaluate errors and rankings on a
  separate DFT test set that keeps related molecular identities in the same split and check selected generated candidates explicitly with DFT.
- Use the DFT predictor in generation after validating its performance. The existing NMO prior can be reused if
  vocabulary and model compatibility are unchanged.

The reference clusters and third plot axis include an xTB-derived property.
Reference and candidate properties must be calculated or predicted on the same
basis: compare xTB with xTB, or build a DFT-based reference for DFT predictions.
The saved descriptor preprocessing and PCA models can be reused if descriptor
calculations remain unchanged.
These are planned steps; the current workflow uses the xTB-trained model.

## Chemical-space reference

Clustering follows the approach discussed with Joe Gilkes and described in
Koczor-Benda et al., [*Generative design of functional organic molecules for
terahertz radiation detection*](https://doi.org/10.1039/D5DD00106D)
(*Digital Discovery*, 2025), which he co-authored: combine bonding and SOAP
structural PCA coordinates with a molecular property, then apply BIRCH followed
by agglomerative clustering. Here it is applied to the combined SMILES/GGS data
using `log_P_upconversion`; the exact inputs are listed below.

All paths in this table are relative to `chemical_space_reference/`.

| File or directory | Purpose |
| --- | --- |
| `smiles_runs/`, `ggs_runs/` | Molecular run data used to construct the reference. |
| `dim.py` | Calculates bonding and structural descriptors, fits PCA and saves the preprocessing settings and fitted PCA models. |
| `dim_reduction_details/` | Descriptor caches, PCA embeddings and fitted PCA data used to construct the reference. |
| `analyse.py` | Clusters molecules using PCA coordinates and `log_P_upconversion` using BIRCH followed by agglomerative clustering, and exports the reference. |
| `cluster_plots/` | Reference-cluster plots and analysis tables. |
| `clustering_feature_scaling.csv` | Centering and scaling applied to each clustering feature. |
| `reference.pkl` | Saved preprocessing settings, PCA models, clustering models and reference molecules used to map candidates. |
| `reference.csv` | Reference molecules, coordinates, properties and cluster labels. |
| `reference.html` | Interactive reference plot used as the background for candidate overlays. |
| `reference.json` | Reference counts, settings and records of the data and methods used. |

### When are PCA and clustering applied?

**PCA is fitted before new candidates are added.** The reference contains 68,268
rows: 36,754 from SMILES runs and 31,514 from GGS runs. Bonding descriptors
characterise molecular connectivity and chemistry; averaged SOAP descriptors
characterise local three-dimensional atomic environments. Each descriptor set
has its own fitted preprocessing and PCA model.

Reference construction then combines three bonding PCs, three structural PCs
and `log_P_upconversion`. These seven features are centred and scaled before
BIRCH and agglomerative clustering produce the 20 final reference clusters.
The property is added after descriptor PCA; it is not an input to either PCA fit.

**After generation**, each candidate's descriptors are calculated using the same
definitions, conformer-generation settings and element handling as the reference.
For chemical-space mapping this includes removing literal `[Au]` from SMILES.
Missing bonding-descriptor values use the saved reference medians, and the saved
descriptor selection, scaling and PCA models are applied with `transform`.

The candidate's predicted property is appended to its six PCA coordinates,
and the saved seven-feature scaling is applied. It receives the final cluster
label of its nearest BIRCH subcluster. Neither the PCA axes nor reference cluster
boundaries are refitted, so adding candidates does not move existing points.

The HTML shows bonding PC1, structural PC1 and the logarithmic property.
It is a three-dimensional view of a seven-dimensional clustering calculation:
points close together in the plot can differ in the other PCs. Candidate
property values are FragNet predictions; reference values retain their xTB basis.
Cluster assignment always selects an existing cluster, even for a distant
candidate; the exported distances help flag candidates outside well-covered
regions, but are not calibrated prediction uncertainties.

## Saved generator and analysis tools

The existing prior and its matching vocabulary are reused for each generation run.

| Path | Purpose |
| --- | --- |
| `nmo_fragnet_generation/generator_prior/prior.pt` | Pretrained RNN parameters: the starting point for a generation run. |
| `nmo_fragnet_generation/generator_prior/prior_manifest.json` | Prior settings and file hashes used to check that the checkpoint and vocabulary match. |
| `generator_prior/valid_ratio.txt`, `valid_ratio.pdf` | Fraction of sampled molecules that were valid during pretraining; inside `nmo_fragnet_generation/`. |
| `generator_prior/n_atoms.txt`, `n_atoms_histogram.pdf` | Atom counts and their distribution for valid sampled molecules; inside `nmo_fragnet_generation/`. |
| `fragnet_analysis/embeddings.py` | Creates an HTML report of FragNet fragment and fragment-connection contributions. |
| `fragnet_analysis/neighbours.py` | Finds similar molecules using embedding cosine similarity or Morgan-fingerprint Tanimoto similarity, depending on the input. |
| `fragnet_analysis/tanimoto_similarity/` | Tables and plots relating predictor error to molecular similarity. |

Analysis tools are independent of the generation/mapping pipeline and accept
explicit input/output paths.

## Dependencies and reproducibility

| Environment | Tasks | Principal dependencies |
| --- | --- | --- |
| `nmo` | Generation and generator pretraining. | NMO dependencies, including PyTorch, RDKit, NumPy, pandas, Matplotlib, tqdm, GGS and group-selfies. |
| `fragnet` | Predictor training, verification, prediction and mapping. | PyTorch, PyTorch Geometric, torch-scatter, RDKit, NumPy, pandas, PyYAML, NetworkX, SciPy, scikit-learn and LMDB. Mapping also requires ASE, DScribe and Plotly. |

NMO runs in `nmo` and sends molecules to a separate process running in `fragnet`.
That prediction process stays open throughout generation to avoid repeatedly
loading the model. The Slurm script selects both environments. Mapping requires
RDKit, scikit-learn and DScribe versions to match those recorded in the reference.

### Environment specifications

Export the working Conda environments on SCRTP and commit the reviewed YAML files:

```bash
conda env export -n nmo | sed '/^prefix:/d' > environment-nmo.yml
conda env export -n fragnet | sed '/^prefix:/d' > environment-fragnet.yml
```

Retain package versions, channels and pip dependencies. Replace installation paths containing a user's home directory with the paths to
those packages in the recipient's checkout. Check that each YAML can recreate a
working environment before sharing it.

Create the environments using YAML files saved in the top-level project directory:

```bash
conda env create -f environment-nmo.yml
conda env create -f environment-fragnet.yml
```

Distribute reviewed, tested YAML exports with the code. Checkpoints, graph
datasets and reference bundles are separate from the environments.

## Run the workflow

The GitHub checkout contains the code. Runs also need the saved generator prior,
FragNet checkpoint, verification graphs/predictions and chemical-space reference
files stored on SCRTP.

```bash
cd /storage/msszkb_grp/msshfg
sbatch generate_and_map.sbatch
```

This CPU pilot compares predictions for five saved graphs and five graphs rebuilt
from the same molecules against the stored training predictions. It then runs
with seed 123 and a budget of 256 evaluations counted by NMO's oracle. The reference and its package versions are checked before generation. Mapping
runs in `fragnet`. Change `CALLS` in the batch script to adjust the budget. Each
run name includes the seed and job ID.

### Run seed

`--seed 123` is an arbitrary default; valid generation seeds are integers from
0 to `2**32 - 1`. Use different seeds for independent generation runs, retaining
the same prior and predictor checkpoint. Set `SEED` in the batch script; the output directory name uses it automatically. The predictor's graph-construction seed is taken
separately from `fragnet_selected.yaml`. Reproducibility also depends on the
software, hardware and execution configuration.

Each job writes to `nmo_fragnet_generation/runs/seed123_jobJOBID/`:

| Output | Contents |
| --- | --- |
| `run.log` | Combined Slurm output and errors, including verification results. |
| `generation/full_history.csv` | NMO search history and fitness values. |
| `generation/fragnet_predictions.csv` | Predictions, failures and oracle-call numbers; also the mapping input. |
| `generation/agent_final.pt` | Final generator parameters. |
| `generation/nmo_fragnet.ini`, `train.py` | Exact generation configuration and working NMO script. |
| `generation/run_summary.json` | Run status, counts, settings and source/model hashes. |
| `generation/fragnet_worker.log` | Predictor-process diagnostics. |
| `space/candidates_in_space.csv`, `candidates_in_space.html` | Unique successful candidates, assigned clusters and an interactive reference overlay. |
| `space/projection_summary.json` | Mapping counts and reference identity. |
| `space/projection_rejected.csv` | Written only when molecules are rejected during mapping. |

The run reuses existing models and reference files. It omits duplicate HDF5 records,
legacy analysis plots and a second candidate CSV. Conformers are calculated in
memory; add `--save-conformers` to the mapping command to retain an SDF.
The HTML embeds the reference plot and Plotly so it can be opened offline; it and
the final generator checkpoint are likely to be the largest outputs.
Recovery checkpoints are retained after an interrupted run and removed after
successful generation; `agent_final.pt` contains model parameters, not the full
optimiser/replay state. Automatic resumption is not provided by this launcher.

Independent scoring and projection, using the `fragnet` environment and a CSV
with a `smiles` column:

```bash
python generate_molecules.py predict --input molecules.csv --output predictions_new
python map_chemical_space.py --input predictions_new/predictions.csv --output mapped_new
```

## Optional generator pretraining

For deliberate retraining, using the `nmo` environment:

```bash
python pretrain_generator.py nmo_fragnet_generation/pretrain.ini --seed 123
```

Output goes to `nmo_fragnet_generation/generator_prior_new`; the script stops if
that directory already contains files.
A `prior_manifest.json` is not created automatically for the new checkpoint.
After checking a new prior, supply its checkpoint path with generation's `--prior`
option. Keep the supplied
prior and its matching manifest together.

Local checks cover syntax, unchanged NMO calculations/configuration, synthetic
PCA/cluster mapping, candidate filtering and checkpoint cleanup on successful
versus interrupted runs. The SCRTP check has also reproduced saved and rebuilt graph predictions for five
molecules (maximum difference about 1.9 × 10⁻⁶). End-to-end generation and mapping
remain to be confirmed by the pilot run.

