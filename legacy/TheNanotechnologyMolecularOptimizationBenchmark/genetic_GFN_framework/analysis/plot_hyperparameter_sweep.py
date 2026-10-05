import sys, os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



if __name__ == '__main__':
    path = sys.argv[1]


    csv_path = os.path.join(path, "all_stats.csv")
    df = pd.read_csv(csv_path)

    run_dirs = df["directory"]
    aucs = df["auc_score"]

    runs = {}
    #indices = {}
    for run, auc in zip(run_dirs, aucs):
        #get everything before _seed
        run_base = run.rsplit("_seed", 1)[0]
        #name is run_3_seed_0 -> 3
        index = int(run.split("_")[1])
        try:
            runs[index] += [auc]
            #indices[run_base] = index
        except KeyError:
            runs[index] = [auc]
            #indices[run_base] = index




    #plot runs as scatter plot with mean and std

    fig, ax = plt.subplots()
    x = np.arange(len(runs))
    means = []
    stds = []
    labels = []
    pos = []
    for i, (run, aucs) in enumerate(runs.items()):
        means.append(np.mean(aucs))
        stds.append(np.std(aucs))
        labels.append(run)
        pos.append(run)


    normalized_means = (np.array(means) - np.min(means)) / (np.max(means) - np.min(means))
    colors = plt.cm.RdYlGn(normalized_means)
    ax.bar(pos, means, color=colors, alpha=0.8)
    ax.errorbar(pos, means, yerr=stds, fmt='o', color='black', capsize=5)


    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    plt.ylim(0.3,1.0)


    plt.xticks(pos, pos, rotation=90)

    plt.savefig(f"{path}/hyperparameter_sweep_scatter.svg")

    #print table of results to console sorted by mean auc
    print(f"{'Run':<20} {'Mean AUC':<10} {'Std AUC':<10}")
    sorted_runs = sorted(runs.items(), key=lambda item: np.mean(item[1]), reverse=True)
    for run, aucs in sorted_runs:
        max_auc = np.max(aucs)
        mean_auc = np.mean(aucs)
        min_auc = np.min(aucs)
        std_auc = np.std(aucs)

        print(f"{run:<20} {mean_auc:<10.4f} {std_auc:<10.4f} (min: {min_auc:.4f}, max: {max_auc:.4f})")

    with open(f"{path}/hyperparameter_sweep_scatter.txt", "w") as f: 

        f.write(f"{'Run':<20} {'Mean AUC':<10} {'Std AUC':<10} \n")
        for run, aucs in sorted_runs:
            max_auc = np.max(aucs)
            mean_auc = np.mean(aucs)
            min_auc = np.min(aucs)
            std_auc = np.std(aucs)

            f.write(f"{run:<20} {mean_auc:<10.4f} {std_auc:<10.4f} (min: {min_auc:.4f}, max: {max_auc:.4f})\n")
