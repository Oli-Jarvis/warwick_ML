# Uploading NMO Results to NOMAD

This directory contains the schema and tooling to publish benchmark results as FAIR datasets on [NOMAD](https://nomad-lab.eu) — a materials-science data repository that assigns DOIs to datasets.

The workflow exports **relevant molecules** (those satisfying the oracle-specific relevance criteria) from the raw HDF5 output files produced by the oracle handler, and uploads them as a single structured, searchable dataset to NOMAD. All three tasks share one upload and one DOI; the `task` field on each entry identifies the subtask.

Only methods that found relevant molecules on **all three tasks** should be submitted.

---

## Relevance criteria

A molecule is considered relevant if it meets the following thresholds (matching `scripts/analyze.py`):

| Task | Criterion |
|---|---|
| Phonon | `k_ph < 0.25` pW/K  **and**  `SA < 4.5` |
| Thermoelectric | `ZT > 3.0` at 300 K  **and**  `SA < 4.5` |
| Optomechanics | `log10(P_upconversion) > 7.88`  **and**  `SA < 4.5` |

---

## Files

```
nomad/
  schema.archive.yaml   ← NOMAD data schema (no installation required)
  README.md             ← this file

scripts/
  export_for_nomad.py   ← step 1: extract relevant molecules → .archive.yaml
  upload_to_nomad.py    ← step 2: upload to NOMAD, create dataset
```

---

## Prerequisites

**Python packages** (beyond the base NMO environment):

```bash
pip install requests pyyaml
```

**NOMAD API token** — generate one at `https://nomad-lab.eu` → *Your account* → *API token*, then export it:

```bash
export NOMAD_API_TOKEN=<your token>
```

---

## Step 1 — Export relevant molecules

Run `export_for_nomad.py` once for all three tasks together. The script searches
recursively for `worker_*.h5` files under each task directory, filters relevant
molecules, and writes one `.archive.yaml` file per molecule into a single flat
output directory.

```bash
python scripts/export_for_nomad.py \
  --phonon_dir            /path/to/results/phonon/               \
  --phonon_config         configs/config_phonon.ini              \
  --thermoelectric_dir    /path/to/results/thermoelectric/       \
  --thermoelectric_config configs/config_thermoelectric.ini      \
  --optomechanics_dir     /path/to/results/optomechanics/        \
  --optomechanics_config  configs/config_optomechanics.ini       \
  --model_name    "GGS"                                          \
  --model_version "1.0.0"                                        \
  --model_url     "https://github.com/..."                       \
  --hyperparams   hyperparams.yaml                               \
  --encoding      GGS                                            \
  --grammar       grammar.txt                                    \
  --output_dir    ./nomad_export/
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--phonon_dir` | yes | Root directory for phonon task results (searched recursively for `worker_*.h5`) |
| `--phonon_config` | no | Config file(s) for the phonon oracle. Default: `<phonon_dir>/config.ini`. Pass multiple files with repeated flags. |
| `--thermoelectric_dir` | yes | Root directory for thermoelectric task results |
| `--thermoelectric_config` | no | Config file(s) for the thermoelectric oracle. Default: `<thermoelectric_dir>/config.ini` |
| `--optomechanics_dir` | yes | Root directory for optomechanics task results |
| `--optomechanics_config` | no | Config file(s) for the optomechanics oracle. Default: `<optomechanics_dir>/config.ini` |
| `--model_name` | yes | Name of the optimization algorithm (e.g. `GGS`, `REINVENT4`, `Graph-GA`) |
| `--model_version` | yes | Version string (e.g. `1.0.0`) |
| `--model_url` | yes | URL to the algorithm repository or paper |
| `--hyperparams` | yes | YAML file with optimizer hyperparameters (shared across all three tasks) |
| `--encoding` | no | Input encoding used by the optimizer (e.g. `GGS`, `SMILES`) |
| `--grammar` | no | Grammar or vocabulary plain-text file used by the optimizer (e.g. GGS grammar file, SMILES token vocabulary). Omit if no explicit grammar/vocabulary is used. |
| `--output_dir` | yes | Destination directory for all `.archive.yaml` files |
| `--verbose` / `-v` | no | Print per-molecule rejection reasons |

### Hyperparameters file

Create a single `hyperparams.yaml` with the optimizer settings used across all three tasks:

```yaml
n_oracle_calls: 10000
batch_size: 64
learning_rate: 0.001
n_steps: 500
# add any further algorithm-specific parameters here
```

### Output

```
nomad_export/
  schema.archive.yaml      ← copied automatically
  <hash>.archive.yaml      ← one file per relevant molecule (all tasks mixed)
  ...
```

Each `.archive.yaml` file contains the full structured record for one molecule: 3D geometry, scalar properties, transmission spectra, calculation provenance, and algorithm metadata. The `task` field identifies which benchmark task the molecule belongs to.

The script prints a per-task summary at the end and warns if any task produced zero relevant molecules.

---

## Step 2 — Upload to NOMAD

```bash
python scripts/upload_to_nomad.py \
  --export_dir   ./nomad_export/           \
  --dataset_name "NMO Benchmark Dataset"   \
  --draft
```

This creates **one NOMAD upload** and **one dataset** (which will receive a single DOI on publication). Use `--draft` to keep everything private for review; remove it to publish immediately.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--export_dir` | — | Flat export directory produced by `export_for_nomad.py` |
| `--dataset_name` | `NMO Benchmark Dataset` | Name of the NOMAD dataset |
| `--draft` | off | Leave upload as draft (recommended; publish via the NOMAD web UI after review) |

### Output

```
--- Summary ---
  upload_id  = abc123
  upload URL = https://nomad-lab.eu/user/uploads/upload/id/abc123
  dataset_id = def456
  dataset URL= https://nomad-lab.eu/user/datasets/dataset/id/def456
```

---

## Alternative: Hugging Face submission via Parquet

Instead of uploading to NOMAD you can convert the `.archive.yaml` files produced by `export_for_nomad.py` into a single Parquet file and push it to the [Hugging Face Hub](https://huggingface.co/datasets).

### Prerequisites

```bash
pip install pyarrow pyyaml numpy
pip install datasets huggingface_hub   # only needed for --push-to-hub
```

Log in to the Hub before pushing:

```bash
huggingface-cli login
```

### Step 2 (alternative) — Convert to Parquet

```bash
python nomad/convert_to_parquet.py \
  --input  "./nomad_export/**/*.yaml" \
  --output nmo.parquet
```

To convert **and** push to the Hub in one command:

```bash
python nomad/convert_to_parquet.py \
  --input      "./nomad_export/**/*.yaml" \
  --output     nmo.parquet               \
  --push-to-hub your-username/nmo-benchmark
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--input` | — | Glob pattern for `.archive.yaml` files produced by `export_for_nomad.py`, e.g. `"./nomad_export/**/*.yaml"` |
| `--output` | — | Output Parquet file path, e.g. `nmo.parquet` |
| `--push-to-hub` | off | Hugging Face dataset repo to push to, e.g. `your-username/nmo-benchmark` |
| `--batch-size` | 500 | Number of YAML files processed per batch before flushing to disk (keeps memory flat) |
| `--workers` | 4 | Parallel worker processes for YAML parsing |

### Column naming convention

The nested NOMAD schema is flattened into columns using the `section__field` convention (double underscore):

| Column | Type | Description |
|---|---|---|
| `task` | string | `phonon`, `thermoelectric`, or `optomechanics` |
| `hash` | string | Unique molecule hash |
| `benchmark_context__model_name` | string | Optimizer name |
| `benchmark_context__model_version` | string | Optimizer version |
| `benchmark_context__model_url` | string | Repository or paper URL |
| `benchmark_context__hyperparameters` | string | Hyperparameter YAML |
| `benchmark_context__grammar` | string | Grammar/vocabulary file (if applicable) |
| `calculation_settings__geometry_method` | string | Geometry optimization method |
| `calculation_settings__temperature` | float | Simulation temperature [K] |
| `calculation_settings__oracle_config` | string | Full oracle config file(s) |
| `molecular_junction__smiles` | string | SMILES with Au-S anchors |
| `molecular_junction__sa_score` | float | Synthetic accessibility score |
| `molecular_junction__hl_gap` | float | HOMO-LUMO gap |
| `molecular_junction__oracle_call` | int | Oracle call index |
| `structure__species` | list[string] | Element symbols of all atoms |
| `structure__positions` | list[list[float]] | 3D coordinates [Å] |
| `phonon_transport__kappa` | float | Phonon thermal conductance κ_ph [pW/K] |
| `phonon_transport__energy` | list[float] | Phonon energy grid [eV] |
| `phonon_transport__transmission` | list[float] | Phonon transmission spectrum |
| `electronic_transport__conductance` | float | Electrical conductance G [G₀] |
| `electronic_transport__seebeck` | float | Seebeck coefficient S [μV/K] |
| `electronic_transport__kappa_el` | float | Electronic thermal conductance [pW/K] |
| `electronic_transport__zt` | float | Figure of merit ZT at 300 K |
| `electronic_transport__energy` | list[float] | Electronic energy grid [eV] |
| `electronic_transport__transmission` | list[float] | Electronic transmission spectrum |
| `upconversion__log_p_upconversion` | float | log₁₀(P_upconversion) |
| `upconversion__log_p_upconversion_scaled` | float | Length-and-area-scaled log₁₀(P) |
| `upconversion__molecular_length` | float | Molecular length [Å] |
| `upconversion__surface_area` | float | Projected surface area [Å²] |

Fields that do not apply to a given task are stored as `null`.

---

## What is stored per molecule

Each NOMAD entry contains the following structured data:

**Benchmark context** — algorithm name, version, URL, hyperparameters (shared across all three tasks), and grammar/vocabulary file (if applicable).

**Calculation settings** — full computational provenance (task-specific):

| Task | Geometry | Additional |
|---|---|---|
| Phonon | GFN1-xTB (`xtb --opt extreme --gfn 1`) | Hessian: GFN1-xTB; Electronic: GFN1-xTB (dxtb); Transport: NEGF |
| Thermoelectric | GFN1-xTB (`xtb --opt extreme --gfn 1`) | Hessian: GFN1-xTB; Electronic: GFN1-xTB (dxtb); Transport: NEGF |
| Optomechanics | GFN2-xTB (`xtb --opt extreme --gfn 2`) | Upconversion: PTB (`xtb --ptb --hess --raman`) |

**Molecular junction** — SMILES (with Au-S anchors), input encoding, SA score, HOMO-LUMO gap, oracle call index, fitness value.

**Structure** — relaxed 3D geometry of the full junction (including Au-S anchors), aligned along the z-axis.

**Task-specific properties:**

| Task | Stored quantities |
|---|---|
| Phonon | κ_ph [pW/K], T_ph(ω) spectrum; electrode: Debye model (E_D = 20 meV, γ = −7 eV/Å²) |
| Thermoelectric | G [G₀], S [μV/K], κ_el [pW/K], ZT at 300 K, T_el(E) and T_ph(ω) spectra; phonon electrode: Debye model; electronic electrode: Au(111) 3×4×6 slab, surface Green's function from DFTB+ |
| Optomechanics | log₁₀(P_upconversion), length-and-area scaled log₁₀(P), molecular length [Å], projected surface area [Å²] |

---

## Schema

The NOMAD schema is defined in `nomad/schema.archive.yaml` as an inline archive schema — no plugin installation is required. The file is automatically copied into the output directory and uploaded alongside the data files so NOMAD can resolve all section references.
