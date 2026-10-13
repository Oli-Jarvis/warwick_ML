# Warwick molecular workflow

**Generate and score molecules, then map them onto the existing chemical space.**
The generator is already pretrained; normal runs reuse `generator_prior/prior.pt`.

## Files

| File | Purpose |
| --- | --- |
| `generate_molecules.py` | NMO generation with FragNet predictions guiding fitness. Also supports `predict` and `verify`. |
| `map_chemical_space.py` | Maps scored molecules onto the saved PCA and clusters, then creates an HTML plot. |
| `pretrain_generator.py` | Generator pretraining, kept for reference or deliberate retraining. |
| `generate_and_map.sbatch` | Runs verification, generation and mapping on Warwick. |
| `nmo_fragnet_generation/pretrain.ini` | Original pretraining settings, with updated paths and a separate output directory. |

Keep the three Python scripts and batch script at the project root. Python paths
are relative to that root. The batch script and pretraining configuration use
`/storage/msszkb_grp/msshfg`; edit that path if installing elsewhere.

## Required directories

| Directory | Contents used by this workflow |
| --- | --- |
| `NMO_source_code` | Generator, vocabulary, genetic search and NMO molecular processing. |
| `fragnet_source_code` | FragNet graph features and model architecture. |
| `fragnet_config_details` | `fragnet_selected.yaml`, `experiment/ft.pt`, and optional `run_manifest.json` for source checks. Verification also needs `graph_data/train.pkl` and `experiment/train_predictions.csv`. |
| `nmo_fragnet_generation` | `generator_prior/prior.pt`, its `prior_manifest.json`, and new run outputs. |
| `chemical_space_reference` | `reference.pkl`, `reference.html`, reference tables/metadata and the reference-building scripts. |
| `initial_predictor_training` | Original FragNet training code; not run during generation. |
| `fragnet_analysis` | Optional embedding and similarity analyses. |

Large data files remain on Warwick: a code-only GitHub clone is insufficient.
Use the existing `nmo` and `fragnet` Conda environments. Mapping requires the
RDKit, scikit-learn and DScribe versions recorded in `reference.pkl`; the batch
uses `fragnet` for mapping and stops if the versions differ.

## Run

On Warwick:

```bash
cd /storage/msszkb_grp/msshfg
sbatch generate_and_map.sbatch
```

The job checks five saved training predictions using both saved and regenerated
graphs. If they match, it runs a CPU pilot with seed 123 and at most 256 oracle
evaluations, then maps the results. The Slurm job ID gives each run a new name.

- `nmo_fragnet_generation/pilot_seed123_JOBID/`: generation history, adapted
  `agent_final.pt`, all FragNet predictions, unique successful candidates,
  configuration, logs and audit summaries.
- `nmo_fragnet_generation/pilot_seed123_JOBID_space/`: `candidates_in_space.html`,
  mapped CSV, generated conformers, rejected rows and projection summary.

To score a separate CSV with a `smiles` column and map its predictions:

```bash
conda activate fragnet
python generate_molecules.py predict --input molecules.csv --output predictions_new
python map_chemical_space.py --input predictions_new/predictions.csv --output mapped_new
```

Choose new output directories. Each Python command provides `--help`.

## How it works

**Generation/prediction:** load the saved three-layer, 512-unit RNN prior and
check its hash and vocabulary. NMO retains its genetic operators, replay
settings, molecular validation, SA and rotatable-bond calculations. A persistent
FragNet subprocess uses its own environment and supplies predictions inside the
search loop. Graph construction, stereochemical checks and model loading retain
the calculations in `initial_predictor_training/run_fragnet.py`.

Fitness uses predicted `log_P_upconversion` with SA and rotatable-bond penalties:
no extra logarithm or length/area penalty. Failed predictions receive zero
reward and remain marked as failures. Existing fixes for empty batches, the
oracle budget, CPU selection and checkpoint saving apply to a per-run copy of
NMO `train.py`. Original NMO sources and the saved prior are not overwritten.

**Mapping:** calculate candidate descriptors, apply the saved preprocessing/PCA,
assign the nearest saved BIRCH subcluster and its agglomerative label, and overlay
candidates on the original HTML plot. PCA and reference clusters are not refitted.
Clustering uses all saved components and the property axis; the plot displays
bonding PC1, structural PC1 and log P. Predictions are not xTB measurements, and
mapping does not establish molecular novelty.

Your previous `dim4.py` and `analyse4.py` are now
`chemical_space_reference/dim.py` and `analyse.py`. They built the reference:
68,268 rows (36,754 SMILES and 31,514 GGS), with 20 final clusters. Normal mapping
reads their saved export. Historical paths in reference metadata are provenance,
not required input locations. Inference likewise uses the supplied model folder
for `ft.pt`, rather than the old paths in the architecture YAML.

## Optional pretraining

**Not needed for the pilot.** To deliberately train another generator:

```bash
conda activate nmo
python pretrain_generator.py nmo_fragnet_generation/pretrain.ini --seed 123
```

The configuration keeps six epochs, batch size 128, learning rate 0.0005, the
original dataset/vocabulary and disabled masking. It writes to
`generator_prior_new` and refuses a nonempty destination. The historical routine
also retains GGS/descriptor options, diagnostic plots and logs.

This routine does not create `prior_manifest.json`. After validating a new prior,
select it with generation's explicit `--prior` option. Keep the existing prior
and its original manifest together; the manifest's source hash records the
historical script, not this cleaned-up copy.

## Checks and limits

Syntax, command-line help, generation settings, guarded NMO source edits and
synthetic fixed-reference mapping were checked. Readability edits preserve the
prediction/mapping executable structure apart from local variable names and
annotations. Pretraining checks compare both encodings' data-loader arguments
and preserve the training, loss, sampling and checkpoint calculations.

Real checkpoint inference, descriptor generation and Slurm execution still need
testing on Warwick; local checks cannot replace a full run with your saved data.
The batch's initial prediction check is a consistency check, not complete
scientific validation of every generated molecule.
