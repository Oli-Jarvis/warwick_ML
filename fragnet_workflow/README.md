# FragNet → NMO → fixed chemical space

Prepared for Oliver Jarvis from the four supplied text attachments and the supplied
oracle_handler.py. This is a runnable implementation with local numerical/adapter
checks; the actual checkpoint and chemistry stack still need verification on Warwick.

## Start here

Extract `fragnet_workflow.zip` under `/storage/msszkb_grp/msshfg`, giving
`/storage/msszkb_grp/msshfg/fragnet_workflow/`. In your existing FragNet environment:

```bash
cd /storage/msszkb_grp/msshfg/fragnet_workflow
python workflow.py verify --count 5
```

Run this in a CPU allocation if required by Warwick's login-node policy. It loads
`fragnet_combined_selected_stereo/experiment/ft.pt`, reconstructs the architecture
from `fragnet_selected.yaml`, and compares both cached and newly built graphs
against five existing training predictions. No training or test-set selection occurs.
A successful result prints maximum prediction differences and the checkpoint hash.

Then test the supplied example molecules:

```bash
python workflow.py predict --input example_history.csv --output first_predictions
```

This produces `predictions.csv`, including xTB values from the input, FragNet values,
residuals, and explicit failures, plus a checkpoint manifest. Use a fresh output
folder for each run. This sample is a formatting/inference smoke test, not an
independent accuracy evaluation: some examples may be training molecules.

## Confirmed inputs and preprocessing

- Selected run: `/storage/msszkb_grp/msshfg/fragnet_combined_selected_stereo`.
- Checkpoint: `experiment/ft.pt` (complete trained weights; no separate pretraining
  checkpoint is needed at inference).
- Config: `fragnet_selected.yaml`; GAT2/FTHead3, 167/167/17 features, four layers,
  four heads, embedding 128, hidden sizes 64/1920/1472/2048, ReLU.
- FragNet source: `/storage/msszkb_grp/msshfg/smiles_baseline2/FragNet`.
- Inference calls the exact supplied training functions for identity, conformers,
  stereo checks, graph creation, and architecture. `training_reference.py` contains
  the supplied training code with its main execution guard removed; importing it
  never starts training. Native source hashes are checked against the run manifest
  when it is available.
- Target is `log_P_upconversion` as stored, with no further logarithm or
  standardisation. The sample confirms this is base 10.
- FragNet retains Au, S, charges, isotopes and specified supported stereo.
  A raw scaffold such as `CCO` is not equivalent to its NMO S–Au anchored molecule.
  For manual predictions supply the anchored SMILES used in the history `smiles`
  column. The NMO adapter performs the existing anchor preparation automatically.
- Training's final CSVs cover accepted train/validation/test graphs; graph-rejected
  molecules are not guaranteed to have predictions.
- Chemical-space descriptors remove literal `[Au]` as in the supplied `dim4.py`,
  while retaining S. This representation is intentionally different from the
  prediction input. The descriptor route has its own fixed seed (42).

## Fit the reference once

### If you already have the dim4 cache

Use your saved `dim_plots/descriptor_cache` and its matching
`dim_plots/combined_labelled_pca_embedding.csv`. This is the preferred route:

```bash
python workflow.py fit-existing \
  --cache-dir /ABSOLUTE/PATH/TO/dim_plots/descriptor_cache \
  --embedding-csv /ABSOLUTE/PATH/TO/dim_plots/combined_labelled_pca_embedding.csv \
  --output /storage/msszkb_grp/msshfg/chemical_space_reference \
  --components 3 --weight 1 --threshold 0.5 --clusters 20
```

This loads the cached `.npy` descriptor matrices without recalculating SOAP or
RDKit descriptors. It fits PCA from the cached matrices only to recover the
previously unsaved transformation, verifies the resulting coordinates against
the saved CSV, and preserves the saved reference coordinates. New molecules
need descriptors calculated only for themselves. If your old complete analysis
CSV is available, pass `--clustered-csv /ABSOLUTE/PATH/TO/birch_agglomerative_clustered_molecules.csv`
to preserve its labels too; in that case candidates take the label of the
nearest saved reference molecule in the same seven-dimensional scaled space.
Otherwise BIRCH/Ward is re-run on the cached coordinates and the fixed centres
are saved. The saved table, cache and optional analysis must refer to identical
rows in the same order. The command checks this through saved PCA coordinates
and molecule alignment, and fails if they differ. This route accepts historical
DFT reference rows if they are present in your saved embedding and cache.

### If the old descriptor cache is missing

Run in the existing chemistry environment containing RDKit, ASE, DScribe, NumPy,
pandas, scikit-learn, matplotlib and Plotly. Reference fitting requires no Torch.
Use explicit roots to avoid accidentally including archived or generated runs:

```bash
python workflow.py fit-reference \
  --input /storage/msszkb_grp/msshfg/smiles_baseline2 /storage/msszkb_grp/msshfg/ggs_2 \
  --output /storage/msszkb_grp/msshfg/chemical_space_reference \
  --components 3 --weight 1 --threshold 0.5 --clusters 20
```

This reads `full_history.csv` recursively beneath those two roots. It consolidates
canonical molecular identities and averages distinct finite log-P targets, as in
your training policy. It uses all usable molecules, without a performance cutoff.
It reports invalid inputs/conformers. These are the existing computed xTB targets;
FragNet predictions are used for new candidates. DFT data are not loaded by this
command.

Your uploaded `dim4.py` saves coordinates/descriptors but does not persist the
full fitted transformation. Without its cache, the command above creates a NEW
reference map once. It does not claim to reproduce the exact coordinates/labels
of an old plot. The new map is then fixed for future candidates. Existing CSV
coordinates alone cannot supply the missing transformation for arbitrary new molecules.

The fit saves SOAP species, RDKit descriptor names/mask, reference imputation
medians, descriptor scalers, PCA models, cluster scaling, BIRCH centres and their
Ward cluster assignments. New candidates do not refit any of these. Version checks
require the same RDKit, DScribe and sklearn versions used for the reference fit.

Clustering reproduces the uploaded `analyse4.py` calculation:

- First three bonding PCs plus first three structural PCs plus log P (7 dimensions).
- Each centred descriptor block is divided by its PC1 standard deviation.
- Property is centred, divided by its own standard deviation, then multiplied by
  the property weight. PC2/3 retain their relative variance.
- BIRCH subclusters → Ward agglomeration of subcluster centres (unweighted, as in
  your script). Default 20 final clusters. If BIRCH produces fewer than 20 centres,
  choose a smaller threshold or fewer final clusters and a fresh output directory.
- Plots show Bonding PC1, Structural PC1 and log P. `--components 1` makes the
  clustering itself use only those three coordinates.

This confirms the uploaded code's method; it does not independently establish
whether every detail matches an external paper or Joe's intended method.

Reference fitting can be expensive: SOAP descriptors are held in memory, following
the existing dim4 approach. Use an appropriate CPU/memory allocation.

## Predict and place new molecules automatically

Create a CSV with a `smiles` header and one complete anchored molecular SMILES per row:

```bash
python workflow.py predict \
  --input candidates.csv \
  --reference /storage/msszkb_grp/msshfg/chemical_space_reference \
  --output candidate_results
```

Outputs: `predictions.csv`, `prediction_manifest.json`, `candidates_in_space.csv`,
`candidates_in_space.html`, and projection rejection records. Each candidate is
assigned to its nearest fixed BIRCH subcluster, then mapped to that subcluster's
Ward label. This is an explicit out-of-sample assignment rule; agglomerative
clustering itself has no native prediction step. A distant molecule still gets
a nearest cluster. Distance is not calibrated model confidence, and these
clustering distances include the predicted property.

If the prediction and plotting packages live in different environments, run
`predict` without `--reference` in FragNet, then switch to the chemistry environment:

```bash
python workflow.py project \
  --input candidate_results/predictions.csv \
  --reference /storage/msszkb_grp/msshfg/chemical_space_reference \
  --output candidate_projection
```

The project command also accepts the NMO adapter's `fragnet_predictions_<pid>.csv`.

## Connect the NMO generator

`install_nmo.py` is tailored to the supplied oracle source. It validates the exact
patch locations and refuses unfamiliar layouts. It backs up the source to
`oracle_handler.py.pre_fragnet`, adds a dispatcher to both SMILES and GGS paths,
and writes a separate configuration derived from your own existing run config.
The original geometry/anchor preparation remains, but the FragNet backend bypasses
the xTB terahertz call. The original metadata writer still runs.

The installer intentionally leaves `calculated_props = SA,N_rot,P_upconversion`
so the existing one-sided S–Au construction is retained; the new
`property_backend = fragnet` setting controls which evaluator is called. This is
more faithful to your Au-retaining training than simply renaming the property.

Before running the generator, expose this directory to its Python process:

```bash
export PYTHONPATH="/storage/msszkb_grp/msshfg/fragnet_workflow${PYTHONPATH:+:$PYTHONPATH}"
```

Your generator process needs both its existing NMO/GGS dependencies and the
FragNet runtime. Do not rerun setup.sh blindly over the working environment.
The adapter runs FragNet on CPU and keeps the model loaded between batches.

First locate your actual active config and generator command:

```bash
cd /storage/msszkb_grp/msshfg/TheNanotechnologyMolecularOptimizationBenchmark
cat submit.sh
rg -n 'python|config|\.ini|\.yaml|\.yml' submit.sh genetic_GFN_framework --glob '*.sh' --glob 'README*'
```

The repository is https://github.com/blaschma/TheNanotechnologyMolecularOptimizationBenchmark
but its public README cannot identify which local config your submit.sh currently
uses. The public documentation lists `NMO/data/configs/config_optomechanics.ini`;
your generator may assemble/override these settings from another configuration.
Send submit.sh and the config it selects before finalising the launch command.

Once the actual complete run config is known, use the following template, replacing
`/ABSOLUTE/PATH/TO/ACTIVE_CONFIG.ini` with that path:

```bash
cd /storage/msszkb_grp/msshfg/fragnet_workflow
python install_nmo.py \
  --oracle /storage/msszkb_grp/msshfg/TheNanotechnologyMolecularOptimizationBenchmark/NMO/NMO/oracle_handler.py \
  --config /ABSOLUTE/PATH/TO/ACTIVE_CONFIG.ini \
  --new-config /storage/msszkb_grp/msshfg/fragnet_workflow/nmo_fragnet.ini \
  --log-dir /storage/msszkb_grp/msshfg/nmo_fragnet_run
```

This defaults to a dry run. Add `--apply` to write the checked patch and new config.
Point the existing generator command at the NEW configuration, with a NEW generator
output directory too. The installer sets NMO's log_dir; your launcher may have a
separate output directory or override it, which needs checking in submit.sh.

### Reward and output semantics

The new config explicitly sets fitness to sigmoid(log P / 5), a positive monotonic
reward suitable for exploring predicted log P, and assigns zero to failed
predictions. This is a chosen initial reward transform, not a recovered formula
from your old run. SA and N_rot are recorded but do not affect this initial reward.
Its scale affects optimisation dynamics and can be adjusted in the new config.
The old length/area-scaled property is NOT invented: the surrogate does not predict
molecular_length or surface_area, and the adapter does not produce
log_P_upconversion_scaled. Do not retain fitness expressions requiring those fields.

The NMO rewards include `log_P_upconversion` and `fragnet_log_P_upconversion`,
`fragnet_prediction_ok`, and `P_upconversion = 10**predicted_logP` for compatible
consumers. They are all surrogate outputs. xTB-derived values are not computed in
this mode. Dedicated per-process CSV logs label predictions and checkpoint hashes.
Graph rejection is logged. Configuration/model-loading failures stop execution.
Existing NMO preprocessing failures continue to use its original failure logging.

After a generation run, project its adapter CSV in the chemistry environment using
the `project` command above. This completes generation → surrogate reward/logging →
fixed-space comparison; automatic invocation after the generator can be added to
your actual submit.sh once its launch command is available. For xTB comparisons,
retain a separate xTB run/config and join by canonical anchored SMILES. Automatic
paired xTB evaluation is not implemented in this first package.

## Checks performed here and remaining gate

Passed Python syntax checks, numerical equivalence to the uploaded clustering
scaling/BIRCH/Ward logic, reference reprojection, fixed-state checks, patching the
actual provided oracle source, and adapter row-alignment/failure handling tests
with a stub predictor. The actual patched base fitness method was exercised to
verify failed candidates receive zero while valid negative log-P predictions
remain distinguishable.

This workspace lacks Torch, RDKit, DScribe, ASE, Plotly and the checkpoint. Full
molecular prediction, plotting and live NMO/HDF5 integration were NOT run here.
The `verify` command and a small NMO smoke run on Warwick are the next gates.
Use your working environments rather than installing arbitrary newer versions.
