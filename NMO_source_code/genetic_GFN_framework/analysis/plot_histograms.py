import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import os
import sys
import pandas as pd

tdc_oracles = ["tdc_albuterol_similarity",
               "tdc_amlodipine_mpo",
               "tdc_celecoxib_rediscovery",
               "tdc_deco_hop" "tdc_drd2",
               "tdc_fexofenadine_mpo",
               "tdc_gsk3b",
               "tdc_isomers_c7h8n2o2",
               "tdc_isomers_c9h10n2o2pf2cl",
               "tdc_jnk3",
               "tdc_median1",
               "tdc_median2",
               "tdc_mestranol_similarity",
               "tdc_osimertinib_mpo",
               "tdc_perindopril_mpo",
               "tdc_qed",
               "tdc_ranolazine_mpo",
               "tdc_scaffold_hop",
               "tdc_sitagliptin_mpo",
               "tdc_thiothixene_rediscovery",
               "tdc_troglitazone_rediscovery",
               "tdc_valsartan_smarts",
               "tdc_zaleplon_mpo",
               ]

# --- 1. SCRIPT TO GENERATE DUMMY DATA ---
# This part creates sample histogram files for demonstration.
# You can replace this section with your own data loading logic.
def create_sample_data(num_files=10, data_points=5000, num_bins=50):
    """Generates several CSV files with histogram data."""
    if not os.path.exists('histogram_data'):
        os.makedirs('histogram_data')

    print(f"Generating {num_files} sample data files in 'histogram_data/' directory...")

    for i in range(num_files):
        # Create different distributions for each file for visual interest
        mean = np.random.uniform(0.1, 0.9)  # Shift the peak
        std_dev = np.random.uniform(0.05, 0.15)  # Change the spread
        data = np.random.normal(mean, std_dev, data_points)

        # Create a histogram from the data
        density, bin_edges = np.histogram(data, bins=num_bins, range=(0, 1), density=True)

        # Prepare data for saving
        bin_starts = bin_edges[:-1]
        bin_ends = bin_edges[1:]

        # Save to a file in the format: bin_start,bin_end,density
        file_path = os.path.join('histogram_data', f'data_{i:02d}.csv')
        np.savetxt(file_path, np.c_[bin_starts, bin_ends, density], delimiter=',')


# --- 2. SCRIPT TO PLOT THE 3D HISTOGRAMS ---
def plot_3d_histograms(csv_list, names, mode, output_path):
    """Reads histogram data from files and creates a 3D bar plot."""

    data_files = csv_list
    print("Data files:", data_files)

    if not data_files:
        print("No data files found")
        return

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    num_files = len(data_files)
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, num_files))
    legend_proxies = []

    for i, filepath in enumerate(data_files):
        try:
            if mode == "dataset":
                data = np.loadtxt(filepath, delimiter=',', skiprows=1)
            elif mode == "agent":
                data = np.loadtxt(filepath, delimiter=',', skiprows = 1, dtype = object)
        except FileNotFoundError as e:
            print(f"File not found: {filepath}")
            continue

        # Extract data
        if mode == "dataset":
            bin_starts = data[:, 0]
            bin_ends = data[:, 1]
            densities = data[:, 2]
        elif mode == "agent":
            d_ = np.array(data[:, 2], dtype=float)
            densities, bin_edges = np.histogram(d_, bins=100, range=(0, 1), density=True)
            bin_starts = bin_edges[:-1]
            bin_ends = bin_edges[1:]


        # Calculate parameters for the 3D bars
        x_pos = (bin_starts + bin_ends) / 2
        y_pos = np.full_like(x_pos, len(legend_proxies) )
        z_pos = np.zeros_like(x_pos)

        # Dimensions of bars
        dx = bin_ends - bin_starts
        dy = 0.8
        dz = densities

        ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors[i], zsort='average', alpha=0.8)
        label = f"{len(legend_proxies)+1}: {names[i]}"
        proxy = mpatches.Patch(color=colors[i], label=label)
        legend_proxies.append(proxy)

    # --- Customize the Plot ---
    ax.set_xlabel('Value')
    ax.set_ylabel('Index')
    ax.set_zlabel('')
    ax.grid(True, which = "both")
    step = 2
    ax.set_yticks(np.arange(0, num_files, step))

    ax.view_init(elev=20, azim=-65)

    ax.legend(handles=legend_proxies, title='Oracles', bbox_to_anchor=(1.15, 1), loc='upper right')
    plt.subplots_adjust(right=0.8)

    plt.tight_layout()
    plt.savefig(f"{output_path}")
    #plt.show()

def plot_historgrams_from_summaray_csv(csv_path, output_path):
    data = np.loadtxt(csv_path, delimiter=',', skiprows = 1, dtype=str)
    print("Data shape:", data.shape)
    #get the header
    header = str(np.loadtxt(csv_path, delimiter=',', max_rows=1, dtype=str))

    print("Header:", type(header), type(str(header)))

    #get the column names
    column_names = header.strip().split()
    #take all columns beginning with tdc_
    for i in range(len(column_names)):
        print(column_names[i])
    tdc_columns = [col for col in column_names if col.startswith("tdc_")]
    print("TDC columns:", tdc_columns)
    #calculate histogram for each column

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    num_files = len(tdc_columns)
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, num_files))
    legend_proxies = []

    for i, col in enumerate(tdc_columns):
        print("Handling column:", col)
        index = column_names.index(col)
        d_ = np.array(data[:, index], dtype= float)
        density, bin_edges = np.histogram(d_, bins=100, range=(0, 1), density=True)
        bin_starts = bin_edges[:-1]
        bin_ends = bin_edges[1:]
        # Calculate parameters for the 3D bars
        x_pos = (bin_starts + bin_ends) / 2
        y_pos = np.full_like(x_pos, len(legend_proxies))
        z_pos = np.zeros_like(x_pos)

        # Dimensions of bars
        dx = bin_ends - bin_starts
        dy = 0.8
        dz = density

        ax.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, color=colors[i], zsort='average', alpha=0.8)
        tmp = column_names[index].replace("tdc_", "")
        label = f"{len(legend_proxies) + 1}: {tmp}"
        proxy = mpatches.Patch(color=colors[i], label=label)
        legend_proxies.append(proxy)

    # --- Customize the Plot ---
    ax.set_xlabel('Value')
    ax.set_ylabel('Index')
    ax.set_zlabel('')
    ax.grid(True, which="both")
    step = 2
    ax.set_yticks(np.arange(0, num_files, step))

    ax.view_init(elev=20, azim=-65)

    ax.legend(handles=legend_proxies, title='Oracles', bbox_to_anchor=(1.15, 1), loc='upper right')
    plt.subplots_adjust(right=0.8)

    plt.tight_layout()
    plt.savefig(f"{output_path}")




if __name__ == "__main__":
    path = sys.argv[1]
    mode = int(sys.argv[2]) #0: dataset, 1: prior, 2: agent

    #select mode
    if mode == 0:
        mode = "dataset"
    elif mode == 1:
        mode = "prior"
    elif mode == 2:
        mode = "agent"
    else:
        raise ValueError("Unknown mode")

    csv_list = []

    if mode == "dataset":
        for oracle in tdc_oracles:
            path_ = f"{path}/{oracle}_dist_dataset.csv"
            csv_list.append(path_)
            names = [oracle.replace("tdc_", "") for oracle in tdc_oracles]
            output_path = f"{path}/{mode}_plot_histograms.svg"
        plot_3d_histograms(csv_list, names, mode, output_path)
    elif mode == "prior":
        path_ = f"{path}/sampled_sequences_prior.txt"
        output_path = f"{path}/{mode}_plot_histograms.svg"
        plot_historgrams_from_summaray_csv(path_, output_path)

    elif mode == "agent":
        for oracle in tdc_oracles:
            path_ = f"{path}/{oracle}/sampled_sequences_final.txt"
            csv_list.append(path_)
            names = [oracle.replace("tdc_", "") for oracle in tdc_oracles]
            output_path = f"{path}/{mode}_plot_histograms.svg"
        plot_3d_histograms(csv_list, names, mode, output_path)
    else:
        raise ValueError("Unknown mode")





