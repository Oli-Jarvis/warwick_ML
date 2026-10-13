"""
Export relevant molecules from NMO HDF5 output files to NOMAD .archive.yaml format.

All three benchmark tasks are exported in a single run into one flat output directory.
Each molecule's .archive.yaml carries a 'task' field (phonon / thermoelectric /
optomechanics) so NOMAD can filter them as subtasks of one dataset.

A method is considered successful only if it found relevant molecules for all three
tasks. The script warns if any task produced zero exported entries.

HDF5 files are discovered recursively under each task directory, e.g.:
  task_dir/worker_0.h5
  task_dir/seed_0/worker_0.h5
  task_dir/seed_1/worker_0.h5

Oracle type is verified against the HDF5 group structure:
  phonon/                        -> phonon task
  phonon/ + electronic/          -> thermoelectric task
  terahertz/                     -> optomechanics task

Relevance criteria (matching analyze.py):
  phonon:         k_ph < 0.25  AND  SA < 4.5
  thermoelectric: ZT > 3.0     AND  SA < 4.5
  optomechanics:  log_P > 7.88 AND  SA < 4.5

Usage:
  python nomad/export_for_nomad.py \
    --phonon_dir            /results/phonon/                  \
    --phonon_config         configs/config_phonon.ini         \
    --thermoelectric_dir    /results/thermoelectric/          \
    --thermoelectric_config configs/config_thermoelectric.ini \
    --optomechanics_dir     /results/optomechanics/           \
    --optomechanics_config  configs/config_optomechanics.ini  \
    --model_name    "GGS"                                     \
    --model_version "1.2.0"                                   \
    --model_url     "https://github.com/..."                  \
    --hyperparams   hyperparams.yaml                          \
    --encoding      GGS                                       \
    --grammar       grammar.txt                               \
    --output_dir    ./nomad_export/                           \
    --legacy_history
"""

import argparse
import glob
import os
import re
import shutil
import sys

import h5py
import pandas as pd
import yaml
from ase.data import chemical_symbols


# ---------------------------------------------------------------------------
# Path redaction
# ---------------------------------------------------------------------------

_ABS_PATH_RE = re.compile(
    r'^('
    r'[/~]'
    r'|[A-Za-z]:[/\\]'
    r'|\.\.([/\\]|$)'
    r')'
)

def _redact_paths(text):
    """Replace identifying filesystem path values with <redacted>."""
    def _replace(m):
        val = m.group(2).strip()
        val_clean = re.split(r'\s+#', val)[0].strip()
        if _ABS_PATH_RE.match(val_clean):
            return m.group(1) + "<redacted>"
        return m.group(0)
    return re.sub(
        r'^(\s*[\w_][\w_]*\s*[=:]\s*)(.+)$',
        _replace,
        text,
        flags=re.MULTILINE,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ZT_FACTOR = 300.0 * 7.748091729e-5   # T * G0  at 300 K, matches analyze.py
_SCHEMA_FILE = os.path.join(os.path.dirname(__file__), ".", "schema.archive.yaml")

TASKS = ("phonon", "thermoelectric", "optomechanics")


# ---------------------------------------------------------------------------
# YAML dumper: numeric lists in flow style to keep files compact
# ---------------------------------------------------------------------------

class _NomadDumper(yaml.Dumper):
    pass

def _list_representer(dumper, data):
    if data and all(isinstance(v, (int, float)) for v in data):
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=False)

_NomadDumper.add_representer(list, _list_representer)

def _dump(doc, path):
    with open(path, "w") as fh:
        yaml.dump(doc, fh, Dumper=_NomadDumper, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)


# ---------------------------------------------------------------------------
# Oracle detection
# ---------------------------------------------------------------------------

def detect_oracle(h5_group):
    """Return 'phonon', 'thermoelectric', or 'optomechanics' from HDF5 group structure."""
    has_ph  = "phonon"     in h5_group
    has_el  = "electronic" in h5_group
    has_thz = "terahertz"  in h5_group
    if has_ph and has_el:
        return "thermoelectric"
    if has_ph:
        return "phonon"
    if has_thz:
        return "optomechanics"
    return None


# ---------------------------------------------------------------------------
# Relevance filter
# ---------------------------------------------------------------------------

def is_relevant(oracle, attrs):
    try:
        sa_raw = attrs.get("SA")
        if sa_raw is None or float(sa_raw) >= 4.5:
            return False

        if oracle == "phonon":
            kph = attrs.get("k_ph")
            if kph is None:
                return False
            return float(kph) < 0.25

        if oracle == "thermoelectric":
            G, S = attrs.get("G"), attrs.get("S")
            kph, kel = attrs.get("k_ph"), attrs.get("k_el")
            if any(v is None for v in (G, S, kph, kel)):
                return False
            denom = float(kph) + float(kel)
            if denom <= 0:
                return False
            return float(G) * float(S) * float(S) / denom * _ZT_FACTOR > 3.0

        if oracle == "optomechanics":
            lp = attrs.get("log_P_upconversion")
            if lp is None:
                return False
            return float(lp) > 7.88

    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return False


# ---------------------------------------------------------------------------
# Geometry extraction
# ---------------------------------------------------------------------------

def read_xyz_geometries(xyz_path):
    """Parse a multi-structure XYZ file into a dictionary of {hash: (species, positions)}."""
    geometries = {}
    if not os.path.exists(xyz_path):
        return geometries
        
    try:
        with open(xyz_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as exc:
        print(f"  [!] Failed to read {xyz_path}: {exc}")
        return geometries

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        try:
            n_atoms = int(line)
        except ValueError:
            i += 1
            continue

        if i + 1 >= len(lines):
            break
            
        comment = lines[i+1].strip()
        hash_match = re.search(r"hash=([a-f0-9]+)", comment)
        if not hash_match:
            i += n_atoms + 2
            continue
            
        hash_key = hash_match.group(1)
        species = []
        positions = []
        
        for j in range(n_atoms):
            idx = i + 2 + j
            if idx >= len(lines):
                break
            parts = lines[idx].split()
            if len(parts) >= 4:
                species.append(parts[0])
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
        
        geometries[hash_key] = (species, positions)
        i += n_atoms + 2
        
    return geometries


def extract_geometry(h5_group, oracle, hash_key=None, legacy_geometries=None):
    """Extract geometry from HDF5, falling back to legacy XYZ files if missing."""
    species, positions = None, None
    
    # Try HDF5 first
    if h5_group is not None:
        group_name = "terahertz" if oracle == "optomechanics" else "phonon"
        grp = h5_group.get(group_name)
        
        if grp is not None and "atomic_numbers" in grp and "positions" in grp:
            nums = grp["atomic_numbers"][:]
            pos  = grp["positions"][:]
            mask = nums > 0
            if mask.any():
                nums = nums[mask].astype(int)
                pos  = pos[mask]
                species = [chemical_symbols[z] for z in nums]
                positions = pos.tolist()
            
    # Fallback to legacy dictionary if missing or empty
    if (species is None or len(species) == 0) and legacy_geometries and hash_key in legacy_geometries:
        species, positions = legacy_geometries[hash_key]

    return species, positions


# ---------------------------------------------------------------------------
# Transport data extraction
# ---------------------------------------------------------------------------

def extract_phonon(h5_group, attrs):
    if h5_group is None:
        return None
    grp = h5_group.get("phonon")
    if grp is None:
        return None
    kappa = float(grp.attrs.get("kappa", attrs.get("k_ph", 0.0)))
    result = {"kappa": kappa}
    if "energy_eV" in grp and "transmission" in grp:
        result["energy"]       = grp["energy_eV"][:].tolist()
        result["transmission"] = grp["transmission"][:].tolist()
    return result


def extract_electronic(h5_group, attrs):
    if h5_group is None:
        return None
    grp = h5_group.get("electronic")
    if grp is None:
        return None
    G   = float(grp.attrs.get("G",    attrs.get("G",    0.0)))
    S   = float(grp.attrs.get("S",    attrs.get("S",    0.0)))
    kel = float(grp.attrs.get("k_el", attrs.get("k_el", 0.0)))
    kph = float(attrs.get("k_ph", 0.0))
    denom = kph + kel
    zt  = G * S * S / denom * _ZT_FACTOR if denom > 0 else 0.0
    result = {"conductance": G, "seebeck": S, "kappa_el": kel, "zt": zt}
    if "energy_fermi_shifted" in grp and "transmission" in grp:
        result["energy"]       = grp["energy_fermi_shifted"][:].tolist()
        result["transmission"] = grp["transmission"][:].tolist()
    return result


def extract_upconversion(attrs):
    result = {}
    for src, dst in [
        ("log_P_upconversion",       "log_p_upconversion"),
        ("log_P_upconversion_scaled", "log_p_upconversion_scaled"),
        ("molecular_length",          "molecular_length"),
        ("surface_area",              "surface_area"),
    ]:
        v = attrs.get(src)
        if v is not None:
            result[dst] = float(v)
    return result or None


# ---------------------------------------------------------------------------
# Archive document builder
# ---------------------------------------------------------------------------

def build_archive(hash_key, oracle, attrs, benchmark_context, calc_settings, h5_group, legacy_geometries=None):
    mj = {"smiles": str(attrs.get("smiles", ""))}
    for key, dest, cast in [
        ("encoding",    "encoding",    str),
        ("SA",          "sa_score",    float),
        ("hl_gap",      "hl_gap",      float),
        ("oracle_call", "oracle_call", int),
        ("fitness",     "fitness",     float),
    ]:
        val = attrs.get(key)
        if val is not None:
            try:
                mj[dest] = cast(val)
            except (TypeError, ValueError):
                pass

    doc = {
        "data": {
            "m_def":                "../upload/raw/schema.archive.yaml#/definitions/sections/NMOEntry",
            "task":                 oracle,
            "hash":                 hash_key,
            "benchmark_context":    benchmark_context,
            "calculation_settings": calc_settings,
            "molecular_junction":   mj,
        }
    }

    species, positions = extract_geometry(h5_group, oracle, hash_key, legacy_geometries)
    if species is not None:
        doc["data"]["structure"] = {"species": species, "positions": positions}

    if oracle in ("phonon", "thermoelectric"):
        ph = extract_phonon(h5_group, attrs)
        if ph:
            ph["electrode_model"] = (
                "Debye model: E_D = 20 meV, gamma = -7 eV/Ang^2. "
                "Ref: Markussen, J. Chem. Phys. 139, 244101 (2013); "
            )
            doc["data"]["phonon_transport"] = ph

    if oracle == "thermoelectric":
        el = extract_electronic(h5_group, attrs)
        if el:
            el["electrode_geometry"] = "Au(111) 3x4x6 slab (left and right)"
            el["electrode_method"] = (
                "Surface Green's function precomputed with DFTB+ "
                "(Slater-Koster parameters for Au). "
                "Electrode data available at "
                "https://huggingface.co/datasets/guise868/electrode_data"
            )
            doc["data"]["electronic_transport"] = el

    if oracle == "optomechanics":
        uc = extract_upconversion(attrs)
        if uc:
            doc["data"]["upconversion"] = uc

    return doc


# ---------------------------------------------------------------------------
# Calculation settings per task
# ---------------------------------------------------------------------------

def build_calc_settings(task, oracle_config):
    if task == "optomechanics":
        return {
            "geometry_method":     "GFN2-xTB (xtb --opt extreme --gfn 2)",
            "upconversion_method": "PTB (xtb --ptb --hess --raman 289.15 785)",
            "temperature":         289.15,
            "oracle_config":       oracle_config,
        }
    # phonon and thermoelectric both use GFN1-xTB geometry + NEGF transport
    return {
        "geometry_method":             "GFN1-xTB (xtb --opt extreme --gfn 1)",
        "hessian_method":              "GFN1-xTB (xtb --hess --gfn 1)",
        "electronic_structure_method": "GFN1-xTB (dxtb)",
        "transport_method":            "NEGF",
        "temperature":                 300.0,
        "oracle_config":               oracle_config,
    }


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_legacy_csv_only(csv_path, task, benchmark_context, calc_settings, output_dir, verbose=False):
    """Fallback processor for directories that have no HDF5 files but contain a full_history.csv."""
    count = 0
    n_not_relevant = 0
    
    csv_dir = os.path.dirname(csv_path)
    legacy_geometries = {}

    # Load geometries from XYZ
    relaxed_files = glob.glob(os.path.join(csv_dir, "relaxed_dump_*.xyz"))
    for xyz_path in relaxed_files:
        legacy_geometries.update(read_xyz_geometries(xyz_path))
        
    if task == "thermoelectric":
        full_files = glob.glob(os.path.join(csv_dir, "full_dump_*.xyz"))
        for xyz_path in full_files:
            legacy_geometries.update(read_xyz_geometries(xyz_path))

    if verbose and legacy_geometries:
        print(f"    Loaded {len(legacy_geometries)} fallback geometries from XYZ files")

    try:
        df = pd.read_csv(csv_path, sep=";", dtype=str)
    except Exception as exc:
        print(f"  [!] Failed to load legacy CSV {csv_path}: {exc}")
        return 0

    hash_col = "hash" if "hash" in df.columns else "hash_values"
    if hash_col not in df.columns:
        print(f"  [!] No valid hash column found in {csv_path}")
        return 0

    for _, row in df.iterrows():
        h = row[hash_col]
        if pd.isna(h) or not h:
            continue
            
        attrs = {k: v for k, v in row.items() if k != hash_col and pd.notna(v) and v != ""}

        if "kappa" in attrs and "k_ph" not in attrs:
            attrs["k_ph"] = attrs.pop("kappa")

        if not is_relevant(task, attrs):
            n_not_relevant += 1
            if verbose:
                print(f"    skip {str(h)[:8]}: "
                      f"SA={attrs.get('SA','?')} "
                      f"k_ph={attrs.get('k_ph','?')} "
                      f"k_el={attrs.get('k_el','?')} "
                      f"G={attrs.get('G','?')} "
                      f"S={attrs.get('S','?')} "
                      f"log_P={attrs.get('log_P_upconversion','?')}")
            continue

        doc = build_archive(h, task, attrs, benchmark_context, calc_settings, None, legacy_geometries)
        _dump(doc, os.path.join(output_dir, f"{h}.archive.yaml"))
        count += 1

    print(f"    not_relevant={n_not_relevant}  exported={count}")
    return count


def process_h5_file(h5_path, task, benchmark_context, calc_settings, output_dir, verbose=False, legacy_mode=False):
    count = 0
    n_no_meta = n_wrong_task = n_not_relevant = 0

    try:
        fh = h5py.File(h5_path, "r")
    except Exception as exc:
        print(f"  [!] Cannot open {h5_path}: {exc}")
        return 0
        
    legacy_data = {}
    legacy_geometries = {}
    
    if legacy_mode:
        h5_dir = os.path.dirname(h5_path)
        csv_path = os.path.join(h5_dir, "full_history.csv")
        worker_base = os.path.basename(h5_path).replace(".h5", "")
        relaxed_xyz = os.path.join(h5_dir, f"relaxed_dump_{worker_base}.xyz")
        full_xyz = os.path.join(h5_dir, f"full_dump_{worker_base}.xyz")

        # Load attributes from CSV using pandas
        if os.path.exists(csv_path):
            try:
                # Read all as strings to avoid converting hashes to numbers or NaNs unexpectedly
                df = pd.read_csv(csv_path, sep=";", dtype=str)
                hash_col = "hash" if "hash" in df.columns else "hash_values"
                
                if hash_col in df.columns:
                    for _, row in df.iterrows():
                        h = row[hash_col]
                        if pd.notna(h):
                            # Keep only non-null and non-empty values, mapping them like a dict
                            legacy_data[h] = {k: v for k, v in row.items() if k != hash_col and pd.notna(v) and v != ""}
                if verbose:
                    print(f"    Loaded legacy properties for {len(legacy_data)} candidates from full_history.csv")
            except Exception as exc:
                print(f"  [!] Failed to load legacy CSV {csv_path}: {exc}")

        # Load geometries from XYZ
        relaxed_geom = read_xyz_geometries(relaxed_xyz)
        full_geom = read_xyz_geometries(full_xyz) if task == "thermoelectric" else {}
        
        # Combine geometries (priority: full_geom > relaxed_geom if both exist)
        legacy_geometries.update(relaxed_geom)
        legacy_geometries.update(full_geom)
        
        if verbose and legacy_geometries:
            print(f"    Loaded {len(legacy_geometries)} fallback geometries from XYZ files")

    with fh:
        all_keys = list(fh.keys())
        if verbose:
            print(f"    groups={len(all_keys)}")

        for hash_key in all_keys:
            grp = fh[hash_key]

            meta = grp.get("metadata")
            if meta is not None:
                attrs = dict(meta.attrs)
            else:
                attrs = {}
                for sub in ("phonon", "electronic", "terahertz"):
                    sg = grp.get(sub)
                    if sg is not None:
                        attrs.update(sg.attrs)
                        
            # Apply legacy data mappings for missing properties
            if legacy_mode and hash_key in legacy_data:
                for k, v in legacy_data[hash_key].items():
                    if k not in attrs or attrs[k] is None:
                        attrs[k] = v

            if "kappa" in attrs and "k_ph" not in attrs:
                attrs["k_ph"] = attrs.pop("kappa")

            if not attrs:
                n_no_meta += 1
                continue

            if detect_oracle(grp) != task:
                n_wrong_task += 1
                continue

            if not is_relevant(task, attrs):
                n_not_relevant += 1
                if verbose:
                    print(f"    skip {hash_key[:8]}: "
                          f"SA={attrs.get('SA','?')} "
                          f"k_ph={attrs.get('k_ph','?')} "
                          f"k_el={attrs.get('k_el','?')} "
                          f"G={attrs.get('G','?')} "
                          f"S={attrs.get('S','?')} "
                          f"log_P={attrs.get('log_P_upconversion','?')}")
                continue

            doc = build_archive(hash_key, task, attrs, benchmark_context, calc_settings, grp, legacy_geometries)
            _dump(doc, os.path.join(output_dir, f"{hash_key}.archive.yaml"))
            count += 1

    print(f"    no_meta={n_no_meta}  wrong_task={n_wrong_task}  "
          f"not_relevant={n_not_relevant}  exported={count}")
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export relevant NMO molecules (all three tasks) to NOMAD .archive.yaml."
    )
    # Per-task directories and config files
    parser.add_argument("--phonon_dir",            required=True,
                        help="Root directory for phonon task results (searched recursively for worker_*.h5).")
    parser.add_argument("--phonon_config",          nargs="+", default=None,
                        help="Config file(s) for the phonon oracle. "
                             "Default: <phonon_dir>/config.ini")
    parser.add_argument("--thermoelectric_dir",     required=True,
                        help="Root directory for thermoelectric task results.")
    parser.add_argument("--thermoelectric_config",  nargs="+", default=None,
                        help="Config file(s) for the thermoelectric oracle. "
                             "Default: <thermoelectric_dir>/config.ini")
    parser.add_argument("--optomechanics_dir",      required=True,
                        help="Root directory for optomechanics task results.")
    parser.add_argument("--optomechanics_config",   nargs="+", default=None,
                        help="Config file(s) for the optomechanics oracle. "
                             "Default: <optomechanics_dir>/config.ini")
    # Shared optimizer metadata
    parser.add_argument("--model_name",    required=True,
                        help="Name of the optimization algorithm (e.g. GGS, REINVENT4).")
    parser.add_argument("--model_version", required=True,
                        help="Version string (e.g. 1.0.0).")
    parser.add_argument("--model_url",     required=True,
                        help="URL to the algorithm repository or paper.")
    parser.add_argument("--hyperparams",   required=True,
                        help="YAML file with optimizer hyperparameters (shared across all tasks).")
    parser.add_argument("--encoding",      default=None,
                        help="Input encoding used by the optimizer (e.g. GGS, SMILES).")
    parser.add_argument("--grammar",       default=None,
                        help="Grammar or vocabulary plain-text file used by the optimizer "
                             "(GGS grammar file, SMILES token vocabulary, etc.).")
    parser.add_argument("--output_dir",    required=True,
                        help="Destination directory for all .archive.yaml files.")
    parser.add_argument("--legacy_history", action="store_true",
                        help="Attempt to find missing molecule attributes and structures in local CSV and XYZ files.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-molecule rejection reasons.")
    args = parser.parse_args()

    # Default each config to <task_dir>/config.ini if not supplied
    for task_attr, dir_attr in [
        ("phonon_config",         "phonon_dir"),
        ("thermoelectric_config", "thermoelectric_dir"),
        ("optomechanics_config",  "optomechanics_dir"),
    ]:
        if getattr(args, task_attr) is None:
            default_cfg = os.path.join(getattr(args, dir_attr), "config.ini")
            setattr(args, task_attr, [default_cfg])

    os.makedirs(args.output_dir, exist_ok=True)

    # Hyperparameters - accept YAML or plain text
    with open(args.hyperparams) as fh:
        raw_hp = fh.read()
    try:
        hp = yaml.safe_load(raw_hp)
        hyperparams_str = yaml.dump(hp, default_flow_style=False) if isinstance(hp, dict) else raw_hp
    except yaml.YAMLError:
        hyperparams_str = raw_hp
    hyperparams_str = _redact_paths(hyperparams_str)

    # Grammar file
    grammar_str = None
    if args.grammar:
        with open(args.grammar) as fh:
            grammar_str = _redact_paths(fh.read())
    elif args.encoding:
        print(f"[WARNING] --encoding {args.encoding} specified but --grammar was not supplied. "
              "Include the grammar/vocabulary file for full reproducibility.")

    benchmark_context = {
        "model_name":      args.model_name,
        "model_version":   args.model_version,
        "model_url":       args.model_url,
        "hyperparameters": hyperparams_str,
    }
    if grammar_str is not None:
        benchmark_context["grammar"] = grammar_str

    # Copy schema once into output directory
    if not os.path.exists(_SCHEMA_FILE):
        print(f"[ERROR] Schema not found at {_SCHEMA_FILE}", file=sys.stderr)
        sys.exit(1)
        
    shutil.copy(_SCHEMA_FILE, os.path.join(args.output_dir, "schema.archive.yaml"))

    # Per-task processing
    task_dirs = {
        "phonon":         args.phonon_dir,
        "thermoelectric": args.thermoelectric_dir,
        "optomechanics":  args.optomechanics_dir,
    }
    task_configs = {
        "phonon":         args.phonon_config,
        "thermoelectric": args.thermoelectric_config,
        "optomechanics":  args.optomechanics_config,
    }

    totals = {}
    for task in TASKS:
        input_dir    = task_dirs[task]
        config_files = task_configs[task]

        parts = []
        for cfg in config_files:
            with open(cfg) as fh:
                raw = fh.read()
            parts.append(f"# {os.path.basename(cfg)}\n{_redact_paths(raw)}")
        oracle_config = "\n\n---\n\n".join(parts)

        calc_settings = build_calc_settings(task, oracle_config)

        # Recursive search for HDF5 files
        h5_files = sorted(glob.glob(os.path.join(input_dir, "**", "worker_*.h5"), recursive=True))
        
        if not h5_files:
            if args.legacy_history:
                csv_files = sorted(glob.glob(os.path.join(input_dir, "**", "full_history.csv"), recursive=True))
                if not csv_files:
                    print(f"[WARNING] No worker_*.h5 or full_history.csv files found under {input_dir} - skipping {task}.")
                    totals[task] = 0
                    continue
                else:
                    print(f"\n[{task}] {input_dir}  ({len(csv_files)} CSV file(s) for legacy mode without HDF5)")
                    task_total = 0
                    for csv_path in csv_files:
                        rel = os.path.relpath(csv_path, input_dir)
                        print(f"  {rel}:")
                        task_total += process_legacy_csv_only(
                            csv_path, task, benchmark_context, calc_settings,
                            args.output_dir, verbose=args.verbose
                        )
                    totals[task] = task_total
                    continue
            else:
                print(f"[WARNING] No worker_*.h5 files found under {input_dir} - skipping {task}.")
                totals[task] = 0
                continue

        print(f"\n[{task}] {input_dir}  ({len(h5_files)} HDF5 file(s))")
        task_total = 0
        for h5_path in h5_files:
            rel = os.path.relpath(h5_path, input_dir)
            print(f"  {rel}:")
            task_total += process_h5_file(
                h5_path, task, benchmark_context, calc_settings,
                args.output_dir, verbose=args.verbose, legacy_mode=args.legacy_history
            )
        totals[task] = task_total

    # Summary
    print(f"\n--- Export summary ---")
    all_ok = True
    for task in TASKS:
        n = totals.get(task, 0)
        flag = "" if n > 0 else "  [WARNING: no relevant molecules]"
        print(f"  {task:16s} {n} entries{flag}")
        if n == 0:
            all_ok = False
    total = sum(totals.values())
    print(f"  {'TOTAL':16s} {total} entries -> {args.output_dir}/")
    if not all_ok:
        print("\n[WARNING] One or more tasks produced no relevant molecules. "
              "Only methods that succeed on all three tasks should be submitted.")

if __name__ == "__main__":
    main()