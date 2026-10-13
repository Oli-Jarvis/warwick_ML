# Warwick molecular workflow

Generate molecules with FragNet fitness, then map them onto the saved chemical
space. The existing generator prior is reused; pretraining is optional.

## Files

| File | Role |
| --- | --- |
| `generate_molecules.py` | Generation and prediction; also provides `predict` and `verify`. Reuses graph/model functions from `initial_predictor_training/run_fragnet.py` without running predictor training. |
| `map_chemical_space.py` | Applies the saved PCA and cluster assignments, then plots candidates. |
| `pretrain_generator.py` | Runs the existing `NMO_source_code/genetic_GFN_framework/pretrain.py` with the three established fixes. |
| `generate_and_map.sbatch` | Checks predictions, generates a pilot and maps its results. |
| `nmo_fragnet_generation/pretrain.ini` | Original pretraining settings and a separate output directory. |

Place the scripts at the project root. Keep these existing inputs:

- `NMO_source_code/` and `fragnet_source_code/`: original implementations and vocabulary.
- `initial_predictor_training/run_fragnet.py`: shared graph/model functions.
- `fragnet_config_details/`: `fragnet_selected.yaml`, `experiment/ft.pt`, and optional `run_manifest.json` for source checks. Verification also uses `graph_data/train.pkl` and `experiment/train_predictions.csv`.
- `nmo_fragnet_generation/generator_prior/`: `prior.pt` and its `prior_manifest.json`.
- `chemical_space_reference/`: `reference.pkl`, `reference.html`, tables and metadata.

Large data stay on Warwick; cloning GitHub code alone is insufficient. Python
paths follow the script location. The batch and INI use `/storage/msszkb_grp/msshfg`;
edit that path if needed. `fragnet_analysis/` remains optional analysis code.

## Run on Warwick

```bash
cd /storage/msszkb_grp/msshfg
sbatch generate_and_map.sbatch
```

The CPU job checks five stored and regenerated graph predictions, then uses seed
123 and at most 256 oracle evaluations. Each run gets its Slurm job ID in the name.
It uses the existing `nmo` and `fragnet` environments. Mapping runs in `fragnet`
and checks RDKit, scikit-learn and DScribe versions against the saved reference.

Results in `nmo_fragnet_generation/`:
- `pilot_seed123_JOBID/`: history, `agent_final.pt`, all predictions, unique successful candidates, settings, logs and audit summaries.
- `pilot_seed123_JOBID_space/`: `candidates_in_space.html`, mapped CSV, conformers, rejected rows and projection summary.

For a separate CSV containing `smiles`, choose fresh output directories:
```bash
conda activate fragnet
python generate_molecules.py predict --input molecules.csv --output predictions_new
python map_chemical_space.py --input predictions_new/predictions.csv --output mapped_new
```

## Scientific behaviour

Generation loads the three-layer, 512-unit RNN prior and checks its hash and
vocabulary. NMO keeps its genetic search, replay, molecular validation, SA and
rotatable-bond calculations. A persistent FragNet process supplies predictions
inside the search loop. Fitness uses predicted `log_P_upconversion` with SA and
rotatable-bond penalties, without another logarithm or length/area penalties.
Failed predictions receive zero reward and retain their failure status.
Existing empty-batch, budget, CPU and checkpoint fixes affect a per-run copy of
NMO `train.py`; original sources and the saved prior are not overwritten.

Mapping calculates descriptors, applies saved preprocessing/PCA, assigns the
nearest BIRCH subcluster's agglomerative label and overlays the original plot.
It never refits PCA or reclusters. Clustering uses all saved components and the
property axis; display axes are bonding PC1, structural PC1 and log P.
Predictions are not xTB measurements; mapping alone does not establish novelty.

The old `dim4.py` and `analyse4.py` are now `chemical_space_reference/dim.py` and
`analyse.py`. They exported 68,268 reference rows (36,754 SMILES; 31,514 GGS) and
20 clusters. Normal mapping reads that export. Old paths in reference metadata
are provenance; inference also ignores old data paths in the architecture YAML
and loads `ft.pt` from the supplied model directory.

## Optional pretraining

Not needed for a normal run:
```bash
conda activate nmo
python pretrain_generator.py nmo_fragnet_generation/pretrain.ini --seed 123
```

Uses the original dataset/vocabulary, six epochs, batch size 128, learning rate
0.0005 and disabled masking. Writes to `generator_prior_new`; refuses a nonempty
output directory. The wrapper retains the previous fixes: respect the selected
GPU/CPU, omit an unused grammar constructor and execute all six epochs. NMO's
training, GGS/descriptor options, diagnostic plots and logs remain in its source.
It does not create a manifest: select a validated new checkpoint using `--prior`.
Keep the existing prior with its original manifest; its hash records the historical script.

Syntax, shared-function equivalence, source patches and synthetic mapping are
checked locally. Full inference, descriptor generation and Slurm execution still
need testing on Warwick. The initial prediction check tests consistency, not the
scientific validity of every generated molecule. Use `--help` for command options.
