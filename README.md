# Warwick molecular generation

The active workflow has two steps: generate molecules using FragNet predictions,
then map the scored molecules onto the saved SMILES/GGS chemical space.
The generator is already pretrained. Normal runs do not train the prior again.

## Files to run

| File | Role |
| --- | --- |
| `generate_molecules.py` | NMO generation and FragNet prediction, including standalone `predict` and `verify` commands. |
| `map_chemical_space.py` | Transform scored candidates into the fixed reference and display them with its original clusters. |
| `pretrain_generator.py` | Historical NMO pretraining routine, retained for reference or deliberate retraining. |
| `generate_and_map.sbatch` | Run verification, generation and mapping on Warwick. |
| `nmo_fragnet_generation/pretrain.ini` | Original pretraining settings with updated input paths and a separate new output directory. |

Keep these three Python scripts and the batch script at the repository root,
alongside the existing directories. Python defaults are relative to that root.
The batch script and pretraining configuration use `/storage/msszkb_grp/msshfg`;
edit those paths if your installation is elsewhere.

## Existing directories and required data

| Directory | Used for |
| --- | --- |
| `NMO_source_code` | Original generator, vocabulary, genetic operators and NMO molecular processing. |
| `fragnet_source_code` | Native FragNet graph features and model architecture. |
| `fragnet_config_details` | `fragnet_selected.yaml` and `experiment/ft.pt`; verification also requires `graph_data/train.pkl` and `experiment/train_predictions.csv`. Keep `run_manifest.json` if available for the additional source checks. |
| `nmo_fragnet_generation` | Existing `generator_prior/prior.pt`, its `prior_manifest.json`, and new run outputs. |
| `chemical_space_reference` | Saved `reference.pkl`, `reference.html`, `reference.csv`, metadata and reference-building scripts. |
| `initial_predictor_training` | Original property-predictor training code. Not invoked during generation. |
| `fragnet_analysis` | Additional embedding and similarity analyses. Not invoked during generation. |

Large checkpoints, graph caches and reference bundles remain on Warwick and are
excluded from GitHub. A code-only clone cannot run without these existing files.
Use the established `nmo` and `fragnet` environments; mapping must use the RDKit,
scikit-learn and DScribe versions recorded in the reference bundle. The batch
script currently uses `fragnet` for mapping and fails if those versions differ.

## Run on Warwick

From the directory containing the scripts and existing data folders:

```bash
cd /storage/msszkb_grp/msshfg
sbatch generate_and_map.sbatch
```

The batch first checks five stored training predictions, using both saved graphs
and regenerated graphs. It stops on a mismatch. It then runs a CPU pilot with
seed 123 and a maximum budget of 256 oracle evaluations. The run name includes
the Slurm job ID, so an earlier run is not overwritten.

Generation outputs go into `nmo_fragnet_generation/pilot_seed123_JOBID/`:
`full_history.csv`, `agent_final.pt`, `fragnet_predictions.csv`,
`unique_fragnet_candidates.csv`, configuration and audit summaries.
Mapping writes `candidates_in_space.html`, `candidates_in_space.csv`, generated
conformers, rejected rows and a projection summary into the corresponding
`pilot_seed123_JOBID_space/` directory.

For an independent supplied CSV containing a `smiles` column:

```bash
conda activate fragnet
python generate_molecules.py predict --input molecules.csv --output predictions_new
python map_chemical_space.py --input predictions_new/predictions.csv --output mapped_new
```

## What the scripts preserve

`generate_molecules.py` loads the existing three-layer, 512-unit RNN prior and
checks its checkpoint hash and vocabulary compatibility. It uses the existing
NMO training loop, genetic mutations/crossovers, replay settings, molecular
validation, SA and rotatable-bond calculations. FragNet runs in a persistent
subprocess using its own environment, while its predictions guide the search
inside the generation loop. Its graph construction, stereochemical checks and
model loading match `initial_predictor_training/run_fragnet.py`.

The existing fitness uses predicted `log_P_upconversion` with SA and
rotatable-bond penalties. It applies no extra logarithm to the prediction and
no molecular-length or area penalty. Failed predictions retain their failure
status and receive zero reward. The previously established guarded edits for
empty batches, the oracle budget, CPU selection and final checkpoint saving
are applied only to a per-run copy of NMO `train.py`.

`map_chemical_space.py` calculates candidate descriptors and applies the saved
preprocessing and PCA transforms. It assigns candidates through the saved
BIRCH subclusters and their agglomerative labels, and overlays candidates on
the original reference HTML. It never refits PCA or reclusters the reference.
Clustering uses all saved components and the property axis; the display shows
bonding PC1, structural PC1 and log P. A predicted property axis is not an xTB
measurement, and mapping alone does not establish molecular novelty.

The earlier `dim4.py` and `analyse4.py` now appear in this repository as
`chemical_space_reference/dim.py` and `chemical_space_reference/analyse.py`.
They build/export the reference. Normal candidate mapping reads that export
without rerunning them. The current metadata records 68,268 reference rows
(36,754 SMILES and 31,514 GGS) and 20 final clusters. Historical absolute paths
in provenance files describe the original run; mapping does not follow them.
The old paths inside `fragnet_selected.yaml` are likewise not used to locate
inference data: the prediction script reads the checkpoint from the supplied
model directory and uses the YAML's architecture settings.

## Pretraining reference

Do not run this for the normal pilot. If deliberately training another prior:

```bash
conda activate nmo
python pretrain_generator.py nmo_fragnet_generation/pretrain.ini --seed 123
```

The configuration retains six epochs, batch size 128, learning rate 0.0005,
the original vocabulary/dataset and disabled masking. It writes to
`generator_prior_new`, refusing a nonempty destination. The pretraining routine
is unchanged; only source discovery and the overwrite guard were added.
This direct historical routine does not create a `prior_manifest.json`.
Use a newly trained checkpoint only after validation, via the generation
script's explicit `--prior` option. Keep the existing prior and its original
manifest together; its historical source hash describes the original script.

## Validation of this reorganisation

Python syntax, command-line help and Bash syntax were checked. Eight graph/model
functions match the current predictor training script exactly. Generation
settings and fitness match the previous integration, and its guarded working-copy
patches apply to the current NMO source. The pretraining function and all mapping
functions remain unchanged from the earlier supplied scripts. A synthetic
fixed-reference test compares coordinates, cluster labels and distances with
the previous implementation to numerical tolerance of 1e-12.

Full checkpoint inference, chemistry descriptors and a Slurm run still need
validation on Warwick: the development environment lacks the chemistry/model
dependencies and the large saved data. The initial batch verification checks
prediction consistency before generation, but is not a complete scientific
validation of every generated molecule.
