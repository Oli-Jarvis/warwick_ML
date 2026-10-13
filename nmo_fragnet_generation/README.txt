FragNet-driven SMILES generation

Generator: 3-layer RNN, d_model=512; vocabulary Voc_adapted.
Pretraining: translated_smiles.smi, six actual epochs by default.
FragNet checkpoint and historical reference are reused.

Fitness = 1/4 * positive_logP * ((10-SA)/9)^2 / (1+exp(2*(N_rot-3.5)))
positive_logP = log10(1+10^clip(predicted_logP+5,-50,80)).
Length and area penalties have weights zero. This is NOT the complete
original xTB geometry-scaled objective. Raw predictions remain in the CSV.

Working source copies fix the pretraining epoch off-by-one, omit an unused
GroupGrammar construction, respect scheduler CUDA_VISIBLE_DEVICES, skip
empty evaluation batches and cap oracle batches at the remaining budget.
The final generator checkpoint is saved before legacy plotting.
Original source files and installed NMO are not edited.

A pilot requests 256 oracle evaluations, including prediction failures.
The step cap may stop a run before this budget; inspect generation_summary.json.
Generated candidates are not guaranteed novel versus existing molecules.
Candidate chemical-space descriptors are calculated; historical descriptors
and PCA/cluster fits are reused.

Pretraining is reused only when its recorded settings/hash match.
Generation restart/resume is not implemented by this launcher.
If a generation directory already exists, inspect it and choose a new --name.
Existing prior configurations from outside this workflow require a compatible
model AND exactly matching vocabulary/token order when using --prior.
