# NMO generation with FragNet

This workflow uses FragNet in place of the xTB upconversion calculation during
NMO generation, then maps the resulting molecules onto the existing SMILES/GGS
chemical-space reference. The aim is to identify promising molecules and the
fragments or fragment combinations behind their predicted performance.

## How FragNet fits into NMO

NMO retains its generator, genetic search, molecular validation, anchoring and
SA/rotatable-bond calculations. Eligible molecules are sent to a separate FragNet
process, which stays loaded throughout the search. It uses the saved checkpoint
and the same graph construction and stereochemistry handling as predictor training.

FragNet returns `log_P_upconversion` directly. Its weights remain fixed during
generation; NMO updates the generator using the resulting fitness. Failed
predictions receive zero fitness and are recorded. No candidate upconversion
calculations are run with xTB or DFT during this search.

For predicted log-property $y$, SA score $s$ and rotatable-bond count $r$, the
configured fitness is

$$
F=\frac14\log_{10}(1+10^c)
\left(\frac{10-s}{9}\right)^2
\frac{1}{1+\exp[2(r-3.5)]},\qquad c=\operatorname{clip}(y+5,-50,80).
$$

This favours higher predicted upconversion, lower SA and fewer rotatable bonds;
length and area penalties are disabled. High predicted property and high fitness
are therefore different rankings. Clustering is subsequent analysis and does
not direct the search.

The **prior** is the saved starting generator. The supplied three-layer RNN
(hidden size 512, maximum sequence length 140) was pretrained on
`translated_smiles.smi` with `Voc_adapted`, minimising negative sequence
log-likelihood without property labels. Settings: six epochs, batch size 128,
initial Adam learning rate 0.0005, masking and descriptor prediction disabled.
An **epoch** is one pass through the training dataset. Normal runs reuse this
prior; changing the generation seed or property predictor does not itself require
pretraining again.

## Predictor and performance

FragNet was trained on combined SMILES/GGS xTB-derived `log_P_upconversion`.
Molecules differing only in stereochemistry stay in the same split. Targets are
the mean of distinct observed values per canonical isomeric SMILES; graphs retain
specified tetrahedral and double-bond stereochemistry.

| Setting | Value |
| --- | --- |
| Architecture | `gat2`; 4 layers, 4 attention heads, embedding dimension 128 |
| Features | 167 atom/fragment features; 17 bond features |
| Regression head | `FTHead3`; widths 64, 1920, 1472, 2048; ReLU |
| Optimisation | Adam; learning rate ≈ 7.9055 × 10⁻⁴; batch size 128; MSE loss |
| Regularisation | Dropout 0.25; no learning-rate scheduler |
| Training | Seed 123; limit 10,000 epochs; stop after 100 epochs without validation improvement |

The selected checkpoint is from **epoch 100**, with the lowest validation MSE.
Test data were not used for selection.

| Split | Molecules | MAE | RMSE | R² | Spearman ρ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 31,766 | 1.104 | 1.590 | 0.934 | 0.961 |
| Validation | 3,970 | 1.329 | 2.032 | 0.805 | 0.916 |
| Test | 3,960 | 1.262 | 1.925 | 0.806 | 0.915 |

Errors are on the logarithmic target scale; Spearman ρ measures ranking agreement.
These results measure agreement with xTB, not DFT or accuracy on newly generated
molecules. KRR was also trialled but performed less well, so FragNet was retained.

## Chemical space and explainability

The reference follows Joe Gilkes's clustering approach, described with co-authors
in [Koczor-Benda et al., *Digital Discovery* (2025)](https://doi.org/10.1039/D5DD00106D).
It contains 68,268 rows from SMILES/GGS runs and 20 final clusters.

**PCA is fitted once to the reference before candidates are added.** Bonding and
SOAP structural descriptors have separate preprocessing/PCA models. Clustering
combines three bonding PCs, three structural PCs and the logarithmic property,
with saved scaling, using BIRCH followed by agglomerative clustering. The property
is added after PCA; it is not used to fit the descriptor PCA models.

New molecules receive the same descriptor preparation, including the reference's
`[Au]` removal and conformer-generation policy. Saved descriptor selection,
missing-value medians, scaling and PCA models are applied without refitting.
Each candidate receives the final cluster label of its nearest BIRCH subcluster.
Reference points and cluster boundaries remain fixed. The plot shows only bonding
PC1, structural PC1 and the property, so visual proximity does not capture all
seven clustering dimensions. Reference properties are xTB-derived; candidate
properties are predictions. Assignment distances help identify distant candidates
but are not calibrated uncertainty estimates.

For high-performing molecules, the analysis should address:

- **Which fragments support the prediction?** `fragnet_analysis/embeddings.py`
  produces an HTML report of signed masking contributions:
  `prediction(full) − prediction(masked)`. Positive values indicate features
  supporting the model's predicted log-property. Fragment-connection contributions
  are included where supported; attention-only scores do not give a direction.
- **Are these recurring chemical patterns?** Compare fragment contributions across
  top candidates and lower-performing molecules, accounting for cluster and
  scaffold similarity. Contributions explain the predicted property, not the
  separate SA/rotatable-bond terms in fitness.
- **How familiar are the candidates to the model?**
  `fragnet_analysis/neighbours.py` compares Morgan fingerprints or supplied
  embeddings. These tools remain exploratory: the previous similarity analyses
  did not provide a reliable guide to prediction error. Developing a useful
  measure is an open next step.

The report script currently reads saved train/validation/test splits; explaining
new generation outputs requires adapting its input handling and checking that its
unmasked predictions match the generation predictor. Explainability is a separate
analysis step, not an automatic pilot output. Masking reveals model sensitivity,
not proof of a causal chemical mechanism; proposed motifs need xTB/DFT validation.

## Files and running

Keep the main scripts at the project root alongside `NMO_source_code/`,
`fragnet_source_code/` and the directories below. Keep
`initial_predictor_training/run_fragnet.py` available: prediction imports its
shared functions. If moving the project, update paths in the batch script and
pretraining INI.

| File or directory | Role |
| --- | --- |
| `generate_molecules.py` | Generation with FragNet; also independent `predict` and `verify` commands. |
| `map_chemical_space.py` | Fixed-reference projection, cluster assignment and plot. |
| `generate_and_map.sbatch` | Verification, generation and mapping in sequence. |
| `pretrain_generator.py`, `nmo_fragnet_generation/pretrain.ini` | Optional creation of a new generator prior. |
| `fragnet_config_details/` | `fragnet_selected.yaml` and `training_settings.json`; `splits/`; prepared `graph_data/`; trained model and evaluation results in `experiment/`. |
| `fragnet_config_details/experiment/` | `ft.pt` is the selected predictor; `last.pt` is the training checkpoint. Metrics, training history and split predictions record evaluation. Verification needs `train_predictions.csv` and `graph_data/train.pkl` from the parent directory. |
| `nmo_fragnet_generation/generator_prior/` | `prior.pt`, matching `prior_manifest.json` and pretraining validity/atom-count diagnostics. |
| `chemical_space_reference/` | `reference.pkl` contains fitted preprocessing/PCA/clustering; CSV/HTML hold reference molecules and plot; JSON records settings. `dim.py` and `analyse.py` construct it from `smiles_runs/` and `ggs_runs/`; caches and plots remain in `dim_reduction_details/` and `cluster_plots/`. |

Code is on GitHub; required checkpoints, graph datasets and reference files are
stored on SCRTP. The batch checks five saved/rebuilt graph predictions against
training results and checks reference compatibility before generation.

```bash
cd /storage/msszkb_grp/msshfg
sbatch generate_and_map.sbatch
```

Set `SEED` and `CALLS` near the top of the batch script. Defaults are **123** and
**256 oracle evaluations**. The seed is arbitrary (0 to `2**32 - 1`); different
seeds use the same prior and predictor. FragNet's graph-construction seed remains
separate in its YAML. Software and hardware also affect reproducibility.

All outputs are under `nmo_fragnet_generation/runs/seed123_jobJOBID/`:

| Location | Contents |
| --- | --- |
| `run.log` | Combined Slurm output/errors and verification results. |
| `generation/` | `full_history.csv`, `fragnet_predictions.csv`, `agent_final.pt`, `nmo_fragnet.ini`, working `train.py`, `run_summary.json` and `fragnet_worker.log`. |
| `space/` | `candidates_in_space.csv`, `candidates_in_space.html`, `projection_summary.json`; `projection_rejected.csv` only if needed. |

Mapping removes duplicate candidate SMILES. No duplicate HDF5 records or legacy
analysis plots are written. Conformers stay in memory unless `--save-conformers`
is supplied. The offline HTML and final checkpoint are the main large outputs.
Recovery checkpoints remain after interruption and are removed after successful
generation; automatic resumption is not provided. The final agent contains model
parameters, not optimiser/replay state.

Independent prediction and mapping, in `fragnet`, for a CSV with a `smiles` column:

```bash
python generate_molecules.py predict --input molecules.csv --output predictions_new
python map_chemical_space.py --input predictions_new/predictions.csv --output mapped_new
```

Optional pretraining, in `nmo`:

```bash
python pretrain_generator.py nmo_fragnet_generation/pretrain.ini --seed 123
```

This writes `nmo_fragnet_generation/generator_prior_new` and refuses a nonempty
destination. After validation, select the new checkpoint with `--prior`; a new
prior manifest is not created automatically. Keep the supplied prior and its
matching manifest together.

## Environments

| Environment | Use and principal dependencies |
| --- | --- |
| `nmo` | Generation/pretraining; NMO dependencies, PyTorch, RDKit, NumPy, pandas, Matplotlib, tqdm, GGS and group-selfies. |
| `fragnet` | Prediction, training and mapping; PyTorch/PyG, torch-scatter, RDKit, NumPy, pandas, PyYAML, NetworkX, SciPy, scikit-learn, LMDB; ASE, DScribe and Plotly for mapping. |

The batch selects the environments. Mapping requires reference-compatible RDKit,
scikit-learn and DScribe versions; equivalent version spellings are accepted.
Export working environments for distribution:

```bash
conda env export -n nmo | sed '/^prefix:/d' > environment-nmo.yml
conda env export -n fragnet | sed '/^prefix:/d' > environment-fragnet.yml
```

Retain pinned dependencies, correct account-specific installation paths, and test
recreation before sharing. From the project directory:

```bash
conda env create -f environment-nmo.yml
conda env create -f environment-fragnet.yml
```

## Next steps

### Similarity and prediction reliability

Previous trials of Morgan-fingerprint Tanimoto similarity, cosine similarity of
FragNet embeddings, and Euclidean distances after PCA of those embeddings yielded
little useful, consistent guidance on prediction error. None has been established
as a reliable way to identify predictions we can trust. The embedding-distance
analysis is separate from the descriptor PCA used for chemical-space mapping.

Find and validate a similarity measure that helps identify when a candidate is
poorly represented by the training data. Assess its relationship to held-out
prediction errors before using it to prioritise generated molecules or interpret
fragment contributions. Chemical similarity alone is not a confidence estimate.

### DFT fine-tuning

Approximately **200 molecules already have DFT data**. Fine-tune the xTB-trained
FragNet on these consistent DFT targets, using grouped validation/cross-validation
for model selection and an independent test set. Compare DFT errors, rankings and
fragment explanations before using the predictor in generation. The NMO prior
can be reused. Since xTB is already fast, approximating DFT is the main motivation;
any speed benefit over xTB should be measured.

DFT predictions require a reference with a consistent property basis because the
property enters clustering. Descriptor PCA models can remain fixed, but DFT
values must not simply replace xTB values in the existing cluster assignment.

Local checks cover unchanged NMO calculations/configuration, synthetic mapping,
filtering and output cleanup. The five-molecule SCRTP prediction check passed
(maximum difference ≈ 1.9 × 10⁻⁶); full generation/mapping and explainability on
new candidates remain to be validated.
