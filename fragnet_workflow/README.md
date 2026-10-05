# Molecular generation with FragNet

The workflow pretrains an RNN generator, generates molecules using the existing
NMO genetic search and FragNet reward, then projects successful molecules into a
saved chemical space. FragNet predictions guide generation during the search.

## Running

Use the existing `nmo` environment for the commands below, on a compute node.
The launcher starts the `fragnet` interpreter for prediction and projection.
Edit `workflow.ini` for paths, run name, evaluation budget and other run defaults.
Paths are relative to the configured root; command-line flags take precedence.
The scientific settings in `run.py:build_config` retain the pilot configuration.

From the repository root:

```bash
conda activate nmo
python fragnet_workflow/run.py prepare
python fragnet_workflow/run.py pretrain --cpu
python fragnet_workflow/run.py generate --name trial_002 --calls 256 --cpu
```

`prepare` checks required files, creates the NMO configuration and working source
copies, and runs the generator import/tokenizer checks. It does not train.
`pretrain` trains six epochs by default, or reuses a prior whose recorded settings
and checksum match. `generate` requires the prior and automatically projects the
successful molecules after generation. `all` performs both stages.
Choose a new run name every time; the launcher does not resume partial generation.
A supplied `--prior` must have the same architecture and vocabulary/token order.

```bash
python fragnet_workflow/run.py all --name trial_003 --cpu
python fragnet_workflow/run.py project --name trial_002
python fragnet_workflow/run.py predict --input molecules.csv --output predictions
python fragnet_workflow/run.py predict --input molecules.csv --output predictions_with_space --with-space
python fragnet_workflow/run.py project --input scored.csv --output projected
```

Prediction input has a `smiles` column (override with `--smiles-column`). Projection
input has `smiles` and `fragnet_log_P_upconversion` (override with
`--property-column`). Command-line CSV/output paths are relative to your shell.
Use a new output directory. Projection of a named run also requires its projection
directory to be absent or empty.

The existing `start_fragnet_generation.py` command still works. `workflow.py`
retains the lower-level prediction, verification and reference-building commands.
The Slurm scripts should be submitted from the repository root. They retain the
Warwick Conda initialisation and `nmo` environment; edit those for other systems.

## Required inputs

- Selected FragNet directory: `fragnet_selected.yaml`, `run_manifest.json`, and
  `experiment/ft.pt`, with the matching `smiles_baseline2/FragNet` implementation.
- Generator framework and its `data/experiments/Voc_adapted` and
  `data/experiments/translated_smiles.smi` files.
- `generator_prior/prior.pt` and its completion manifest, or a pretraining run.
- `chemical_space_reference/reference.pkl` for the current automatic projection.

Run `git lfs pull` after cloning to retrieve the model, dataset and reference
contents. Small LFS pointer files are not usable model files. The default launcher
checks the reference and dataset even when a generator prior already exists.
Use the same RDKit, scikit-learn and DScribe versions used to fit the reference;
projection explicitly checks them.

## Scientific behaviour

The RNN has three layers and hidden size 512. Generation retains the existing
replay updates, mutation/crossover operations, filters, S-Au anchoring and oracle
budget handling. The property is predicted `log_P_upconversion`; no extra
logarithm is applied to the model output.

Fitness retains the positive log-P transform with offset 5, the SA penalty and
the rotatable-bond penalty. Length and area weights are zero. Predictions and
fitness remain different quantities; failed predictions receive zero fitness.

Candidate descriptors are calculated and transformed using the saved reference.
There is no PCA or cluster refit during projection. The uploaded reference uses
three PCs per descriptor family plus log-P, 20 final clusters and nearest-BIRCH-
subcluster assignment. Its manifest says `original_clusters_preserved: false`;
this refactor does not replace those labels with the earlier clustering.

The guarded changes applied to working generator copies remain unchanged:
CUDA allocation handling, six actual pretraining epochs, empty-batch handling,
remaining-budget limits and saving the final agent before optional plots.

## Files to read

| File | Responsibility |
| --- | --- |
| `workflow.ini` | Paths and routine run settings |
| `run.py` | Generation stages and common command-line entry point |
| `nmo_fragnet_runner.py` | Persistent FragNet subprocess and NMO reward connection |
| `predictor.py` | Load the trained model and predict an aligned batch |
| `model_utils.py` | Molecule identity, stereo checks, graph and model construction |
| `space.py` | Reference transformations, assignment and plotting |
| `legacy_dim.py` | Existing conformer construction used by the space module |

`training_reference.py` now only provides compatibility imports. The original
`run_fragnet.py` training implementation is unchanged, including checkpoint-resume
fingerprints. `install_nmo.py` and `nmo_bridge.py` are the earlier integration
route; do not install that patch when using the current subprocess launcher.
Historical reference setup and integration notes are in `REFERENCE_SETUP.md`.

## Validation before adopting the branch

The standard-library checks compare the relocated functions, stage ordering,
fitness and generated NMO settings with commit
`b240b788c8024c46a46bf365dc9be984fafef440`. They also test paths, CLI overrides,
subprocess routing and protection of existing runs:

```bash
python -m unittest discover -s fragnet_workflow/tests -v
```

On an allocated compute node, use your working FragNet environment for the actual
model/reference comparison. This creates both results in a new directory:

```bash
conda activate fragnet
python fragnet_workflow/check_equivalence.py --count 5 --output validation_refactor
```

This compares regenerated predictions, projected coordinates, cluster labels and
rejection rows, including a duplicate and an invalid SMILES. It also compares the
valid predictions against the saved training predictions and checks that the
model and reference files did not change. The output `comparison.json` is written
only after all comparisons pass. Logs for each version are retained on failure.
This does not exercise a full generator run; use a fresh small generation in the
`nmo` environment after the comparison passes. The original `workflow.py verify`
also tests saved graph tensors, but needs the separately stored `graph_data/train.pkl`.

On Warwick, `sbatch validate.sbatch` from the repository root runs both checks in
the existing `fragnet` environment. It writes `validate-JOBID.log` and a fresh
`validation_refactor_JOBID` directory. The real molecular comparison must pass
before adopting this branch; it could not be executed in the refactoring workspace.

## Later DFT fine-tuning

Keep the current xTB model and split records. DFT fine-tuning should load the full
xTB `ft.pt` into the same architecture and write a separate model directory.
That training mode is not implemented by this refactor. The existing training
script starts fresh runs from the original FragNet pretraining checkpoint.
Using a future DFT predictor with the existing xTB-valued reference also requires
an explicit decision about the property coordinate; it is not a formatting change.
