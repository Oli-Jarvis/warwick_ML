# Molecular generation workflow

Three scripts follow the scientific stages. FragNet predictions guide the
molecular search inside step 2; projection is a separate operation.

| Step | Script | Input | Main output |
| --- | --- | --- | --- |
| 1. Pretrain the generator | `pretrain_generator.py` | SMILES dataset and vocabulary | RNN `prior.pt` and completion manifest |
| 2. Generate and score | `generate_molecules.py` | RNN prior and trained FragNet model | Scored molecules in `fragnet_predictions.csv`, history and final generator |
| 3. Project into chemical space | `project_molecules.py` | Scored molecules and saved reference | `candidates_in_space.csv` and interactive HTML |

## Run the steps

From the repository root, on an allocated compute node:

```bash
conda activate nmo
python fragnet_workflow/pretrain_generator.py --cpu
python fragnet_workflow/generate_molecules.py --name trial_002 --calls 256 --cpu
python fragnet_workflow/project_molecules.py --name trial_002
```

The generation script keeps a FragNet worker in your existing `fragnet` Conda
environment. The projection script also starts that interpreter, so all three
commands can be launched from `nmo`. Run only the stages you need: reuse the prior
by starting at step 2, or repeat projection under a new output location via the
lower-level `workflow.py project` command.

On Warwick, submit each stage separately from the repository root:

```bash
sbatch pretrain.sbatch
# After pretraining succeeds:
sbatch generate.sbatch trial_002
# After generation succeeds:
sbatch project.sbatch trial_002
```

These jobs do not automatically depend on one another. Wait for each stage to
finish successfully before submitting the next. The Slurm files retain your
Warwick Conda initialisation. Generation no longer starts projection automatically.

Edit `fragnet_workflow/workflow.ini` for paths and routine defaults. Relative paths
in that file follow its configured root; CLI paths follow your current directory.
Use `--help` on each script for its own options. The RNN, genetic-search, replay
and fitness settings remain together in `generator.py:build_config`.

## Inputs and reuse

Run `git lfs pull` after cloning to retrieve the actual weights, dataset and
reference. LFS pointers alone cannot be used for prediction.

Step 1 needs the original generator framework, `Voc_adapted` and
`translated_smiles.smi`. It trains six epochs by default. An existing prior is
reused only when its saved settings and checksum match. If relocating a verified
prior, use `--prior /path/to/prior.pt`; it must match the architecture and exact
vocabulary/token order. FragNet and the chemical-space reference are not needed
for this stage.

Step 2 additionally needs the matching FragNet implementation and the selected
model directory (`fragnet_selected.yaml`, `run_manifest.json`, `experiment/ft.pt`).
Every generation needs a fresh name. Partial generations are not resumed.
The oracle budget includes failed predictions; the step limit can end a run early.
Generation output is under `nmo_fragnet_generation/NAME/` by default.

Step 3 needs that run's `fragnet_predictions.csv` and
`chemical_space_reference/reference.pkl`. It filters failures and keeps the first
successful row for each canonical anchored SMILES. Results go to `NAME_space/`.
The summary uses the requested budget from the saved run configuration when
available. It does not retrain the predictor or refit the reference.
Use the same RDKit, scikit-learn and DScribe versions as the reference fit.

## What remains unchanged

- Three-layer RNN with hidden size 512, pretraining settings and tokenisation.
- Replay updates, genetic mutation/crossover, filters and S-Au anchoring.
- FragNet model/graph construction and predicted `log_P_upconversion`.
- Positive log-P fitness transform with offset 5, SA and rotatable-bond penalties;
  length and area weights remain zero. Failed predictions receive zero fitness.
- Candidate descriptors, saved PCA transformations and cluster assignment.
- Guarded generator source fixes for allocated GPUs, actual epoch count, empty
  batches, remaining evaluation budget and saving before optional plots.

The current reference has three PCs per descriptor family plus log-P, 20 final
clusters and nearest-BIRCH-subcluster assignment. Its manifest records
`original_clusters_preserved: false`; no older cluster labels are substituted.

## Supporting code

| File | Purpose |
| --- | --- |
| `generator.py` | Shared generator settings, prepared source copies and process helpers |
| `nmo_fragnet_runner.py` | Persistent FragNet worker and NMO fitness connection |
| `predictor.py`, `model_utils.py` | Load the model, build graphs and predict |
| `space.py`, `conformers.py` | Describe molecules, transform coordinates and assign clusters |
| `settings.py`, `workflow.ini` | Paths and stage-specific command-line settings |
| `workflow.py` | Optional standalone prediction, verification and reference-building utilities |

There is no combined `run.py` or `start_fragnet_generation.py` entry point.
Obsolete integration and analysis scripts are retained under `archive/` for
reference. Historical setup notes remain in `REFERENCE_SETUP.md`.

## Check against the original

```bash
python -m unittest discover -s fragnet_workflow/tests -v
```

These standard-library checks compare moved functions and configuration with
commit `b240b788c8024c46a46bf365dc9be984fafef440`, and exercise stage separation,
file checks, deduplication, failure handling and protection of existing runs.

For actual prediction and projection comparisons, submit `sbatch validate.sbatch`
on Warwick. It writes `validate-JOBID.log` and `validation_refactor_JOBID/`.
The comparison includes a duplicate and an invalid SMILES, checks saved training
predictions, projected coordinates and labels, and verifies that the weights and
reference did not change. `comparison.json` is written only if all checks pass.
The molecular comparison has not been run in the refactoring workspace, which
lacks the required chemistry/ML packages. It does not test a full generation run.

## Predictor training

The original top-level `run_fragnet.py` and its resume fingerprints are unchanged.
Keep the xTB model and split records. Later DFT fine-tuning should initialise from
the full xTB `ft.pt` and write a separate model directory; that mode is a separate
change, not implemented here. Using DFT-valued predictions in an xTB-valued chemical
space also needs an explicit scientific decision about the property coordinate.
