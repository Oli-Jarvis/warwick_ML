import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
import argparse
from sklearn.metrics import auc

# -- Relevance criteria (per oracle) -----------------------------------------

SKIP_DIRS = {"priors", "fritz_logs", "slurm_logs", ".git", "__pycache__", "exported_data", "seed_filtered_xyz", "eval", "GFlow_Mol"}

def _relevant_phonon(df):
    return (df["k_ph"].astype(float) < 0.25) & (df["SA"].astype(float) < 4.5)

def _relevant_thermoelectric(df):
    G, S = df["G"].astype(float), df["S"].astype(float)
    k_ph, k_el = df["k_ph"].astype(float), df["k_el"].astype(float)
    factor = 300 * 7.748091729E-5
    ZT = G * S * S / (k_ph + k_el) * factor
    return (ZT > 3.0) & (df["SA"].astype(float) < 4.5) 

def _relevant_optomechanics(df):
    return (df["log_P_upconversion"].astype(float) > 7.88) & (df["SA"].astype(float) < 4.5) 

RELEVANCE_FN = {
    "phonon": _relevant_phonon,
    "thermoelectric": _relevant_thermoelectric,
    "optomechanics": _relevant_optomechanics,
}


def count_relevant(df, oracle):
    """Return the number of candidates satisfying the oracle-specific relevance criterion."""
    if oracle is None or oracle not in RELEVANCE_FN:
        return None
    try:
        mask = RELEVANCE_FN[oracle](df)
        return int(mask.sum())
    except KeyError as e:
        print(f"  [!] Column {e} not found – skipping relevance check.")
        return None


# -- AUC / statistics --------------------------------------------------------

def calculate_auc_top_x(final_history, top_x):
    oracle_calls = np.sort(final_history["oracle_call"].unique())

    if len(oracle_calls) > 10_000:
        oracle_calls = oracle_calls[:10_000]

    oracle_calls_max = int(np.max(oracle_calls))
    
    # Validation check for range completeness
    expected_sum = oracle_calls_max * (oracle_calls_max + 1) // 2
    actual_sum = int(np.sum(oracle_calls))
    if actual_sum != expected_sum:
        print(f"  [!] Note: Oracle calls range has gaps. Expected sum {expected_sum}, got {actual_sum}.")

    mean_fitness_values = []
    std_fitness_values  = []
    best_fitness_values = []
    mean_sa_values      = []

    for step in oracle_calls:
        cumulative = final_history[final_history["oracle_call"] <= step]
        top_df = cumulative.nlargest(top_x, "fitness")

        mean_fitness_values.append(top_df["fitness"].mean())
        std_fitness_values.append(top_df["fitness"].std(ddof=0))
        best_fitness_values.append(cumulative["fitness"].max())
        mean_sa_values.append(top_df["SA"].mean() if "SA" in top_df.columns else 0.0)

    mean_fitness = np.array(mean_fitness_values)
    std_fitness  = np.array(std_fitness_values)
    best_fitness = np.array(best_fitness_values)
    mean_sa      = np.array(mean_sa_values)

    auc_value = auc(oracle_calls, mean_fitness) / len(oracle_calls)

    return auc_value, mean_fitness, std_fitness, best_fitness, mean_sa, oracle_calls


# -- H5 recovery -------------------------------------------------------------

def analyze_run(h5_filepath, top_x=10, output_dir=".", oracle=None):
    records = []
    print(f"\n--- Analyzing {h5_filepath} ---")

    try:
        with h5py.File(h5_filepath, 'r', libver='latest') as f:
            def collect_metadata(name, obj):
                try:
                    # Identify the 'metadata' group which holds the molecular properties
                    if isinstance(obj, h5py.Group) and name.endswith("metadata"):
                        # Capture all attributes: SMILES, fitness, conductance, etc.
                        meta = dict(obj.attrs)
                        
                        if "oracle_call" in meta and "fitness" in meta:
                            # Extract hash from the parent group name
                            meta["hash"] = obj.parent.name.strip("/")
                            
                            # Standardize numeric types
                            meta["oracle_call"] = int(meta["oracle_call"])
                            meta["fitness"] = float(meta["fitness"])
                            meta["SA"] = float(meta.get("SA", 0.0))
                            
                            records.append(meta)
                except Exception:
                    pass

            try:
                f.visititems(collect_metadata)
            except Exception as e:
                print(f"  [!] visititems interrupted ({e}). Processing {len(records)} found records.")

    except Exception as e:
        print(f"  [X] Could not open HDF5 file: {e}")
        return None

    if not records:
        print(f"  [!] No valid records found in {h5_filepath}.")
        return None

    print(f"  [+] Recovered {len(records)} raw records.")

    df = pd.DataFrame(records)
    df = df[df["oracle_call"] < 10_000]

    if df.empty:
        print(f"  [!] No entries within oracle range < 10,000 in {h5_filepath}.")
        return None

    # Keep best fitness if duplicates exist for the same oracle call
    df = df.sort_values("fitness", ascending=False).drop_duplicates("oracle_call")

    max_oc = df["oracle_call"].max()
    reached_10k = max_oc >= 9_999

    if not reached_10k:
        print(f"  [!] WARNING: reached call {max_oc} (expected 9999).")

    # Create full index to ensure continuous data for AUC and plotting
    full_range = pd.DataFrame({"oracle_call": np.arange(0, max_oc + 1)})
    df_full = full_range.merge(df, on="oracle_call", how="left").sort_values("oracle_call")

    # Fill numeric gaps with 0.0 and string gaps with "none"
    for col in df_full.columns:
        if df_full[col].dtype.kind in 'iufc':
            df_full[col] = df_full[col].fillna(0.0)
        else:
            df_full[col] = df_full[col].fillna("none")

    # Reorder columns: oracle_call and hash first
    cols = df_full.columns.tolist()
    if "hash" in cols:
        cols.insert(1, cols.pop(cols.index("hash")))
        df_full = df_full[cols]

    stem = os.path.splitext(os.path.basename(h5_filepath))[0]
    os.makedirs(output_dir, exist_ok=True)

    # Save detailed CSV
    csv_path = os.path.join(output_dir, f"{stem}_fitness.csv")
    df_full.to_csv(csv_path, index=False, sep=";")
    print(f"  [OK] CSV saved -> {csv_path}")

    # AUC and Stats
    auc_val, mean_fit, std_fit, best_fit, mean_sa, oc = calculate_auc_top_x(df_full, top_x)
    
    top_x_df = df_full[df_full["fitness"] > 0].nlargest(top_x, "fitness")
    m_top_fit = top_x_df["fitness"].mean()
    s_top_fit = top_x_df["fitness"].std(ddof=0)
    m_top_sa = top_x_df["SA"].mean()

    stats_path = os.path.join(output_dir, f"{stem}_auc_top{top_x}.txt")
    np.savetxt(
        stats_path,
        np.column_stack((oc, mean_fit, std_fit, best_fit, mean_sa)),
        header=(
            f"AUC top-{top_x}: {auc_val:.6f} | Global Mean: {m_top_fit:.6f} +/- {s_top_fit:.6f} | "
            f"Max OC: {max_oc}\noracle_call\tmean_fitness\tstd_fitness\tbest_fitness\tmean_SA"
        ),
        fmt="%.6f", delimiter="\t"
    )

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    tag = "" if reached_10k else " (INCOMPLETE)"
    
    ax1.plot(oc, mean_fit, color="crimson", label=f"Mean top-{top_x}")
    ax1.fill_between(oc, mean_fit - std_fit, mean_fit + std_fit, alpha=0.2, color="crimson")
    ax1.plot(oc, best_fit, color="navy", linestyle="--", label="Best (top-1)")
    ax1.set_ylabel("Fitness")
    ax1.set_title(f"{stem}{tag} | AUC: {auc_val:.3f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(oc, mean_sa, color="darkorange", label="Mean SA")
    ax2.set_ylabel("SA Score")
    ax2.set_xlabel("Oracle Call")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{stem}_auc_top{top_x}.svg"))
    plt.close()

    # Relevance count (oracle-specific)
    n_relevant = count_relevant(df_full[df_full["fitness"] > 0], oracle)
    #if n_relevant is not None:
    #    desc = RELEVANCE_CRITERIA[oracle]["description"]
    #    print(f"  [+] Relevant candidates ({desc}): {n_relevant}")

    return {
        "file": stem, "auc": auc_val, "mean_topx_fitness": m_top_fit,
        "std_topx_fitness": s_top_fit, "mean_topx_sa": m_top_sa,
        "max_oracle_call": max_oc, "complete": reached_10k,
        "n_relevant": n_relevant
    }

# -- Entry point -------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_x", type=int, default=10)
    parser.add_argument("--input_dir", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--oracle", type=str, default=None,
                        choices=list(RELEVANCE_FN.keys()),
                        help="Oracle type for relevance metric evaluation")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.input_dir

    h5_files = glob.glob(os.path.join(args.input_dir, "*.h5"))
    if not h5_files:
        print("No .h5 files found.")
    else:
        print(f"Found {len(h5_files)} files: {h5_files}")
        summary_rows = []
        for filepath in sorted(h5_files):
            res = analyze_run(filepath, top_x=args.top_x, output_dir=args.output_dir, oracle=args.oracle)
            if res: summary_rows.append(res)

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows).sort_values(["complete", "auc"], ascending=[False, False])

            # --- Aggregate statistics across all files ---
            n = len(summary_df)
            mean_auc = summary_df["auc"].mean()
            std_auc  = summary_df["auc"].std(ddof=1) if n > 1 else 0.0

            mean_of_means = summary_df["mean_topx_fitness"].mean()
            # Propagated std: Var_total = E[Var_within] + Var_between
            var_within  = (summary_df["std_topx_fitness"] ** 2).mean()
            var_between = summary_df["mean_topx_fitness"].var(ddof=1) if n > 1 else 0.0
            std_propagated = np.sqrt(var_within + var_between)

            agg_row = {
                "file": "AGGREGATE",
                "auc": mean_auc,
                "mean_topx_fitness": mean_of_means,
                "std_topx_fitness": std_propagated,
                "mean_topx_sa": summary_df["mean_topx_sa"].mean(),
                "max_oracle_call": summary_df["max_oracle_call"].max(),
                "complete": summary_df["complete"].all(),
                "n_relevant": summary_df["n_relevant"].sum() if summary_df["n_relevant"].notna().any() else None,
            }
            summary_df = pd.concat([summary_df, pd.DataFrame([agg_row])], ignore_index=True)

            summary_path = os.path.join(args.output_dir, f"summary_top{args.top_x}.csv")
            summary_df.to_csv(summary_path, index=False, sep=";")
            print(f"\n--- Final Summary Table ---")
            print(summary_df.to_string(index=False))
            print(f"\n  Mean AUC: {mean_auc:.6f} +/- {std_auc:.6f}")
            print(f"  Mean top-{args.top_x} fitness: {mean_of_means:.6f} +/- {std_propagated:.6f}")
