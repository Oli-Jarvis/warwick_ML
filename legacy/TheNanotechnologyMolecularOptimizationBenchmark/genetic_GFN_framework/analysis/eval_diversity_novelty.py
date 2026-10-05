import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pandas as pd
from tdc.chem_utils.evaluator import novelty, diversity



import numpy as np

def load_stats_file(path, top_x = None):
    if not top_x:
        data = np.loadtxt(path, delimiter=',', skiprows=1, dtype=str, comments=None)
    else:
        data = np.loadtxt(path, delimiter=',', skiprows=1, max_rows=top_x, dtype=str, comments=None)
    print("Data shape:", data.shape)
    header = str(np.loadtxt(path, delimiter=',', max_rows=1, dtype=str))

    # get the column names
    column_names = header.strip().split()
    smiles_col = [i for i, col in enumerate(column_names) if col == "smiles"]
    assert len(smiles_col) == 1, smiles_col
    smiles = data[:, smiles_col[0]]

    return smiles

def load_top_x_from_full_history(path, top_x):
    #load csv using pandas
    df = pd.read_csv(path, delimiter=',')
    #get the top_x rows based on the "fitness" column
    df_top = df.nlargest(top_x, 'fitness')
    smiles = df_top['smiles'].tolist()

    return smiles



if __name__ == '__main__':
    """
    Usage: python eval_diversity_novelty.py <dataset_stats_path> <full_history_path> [<sampled_stats_path>]
    Providing the sampled_stats_path will compute diversity and novelty for the sampled set as well -> takes very long
    """

    dataset_stats_path = sys.argv[1]
    full_history_path = sys.argv[2]

    full = False
    if len(sys.argv) > 4:
        full = True
        sampled_stats_path = sys.argv[3]


    dataset_smiles = load_stats_file(dataset_stats_path)
    print("Loaded dataset smiles:", len(dataset_smiles))

    if full:
        sampled_smiles = load_stats_file(sampled_stats_path)
        print("Loaded sampled smiles:", len(sampled_smiles))

        sample_novelty = novelty(sampled_smiles, dataset_smiles)

        dataset_diversity = diversity(dataset_smiles)
        sample_diversity = diversity(sampled_smiles)

        print(f"Dataset size: {len(dataset_smiles)}, Sampled size: {len(sampled_smiles)}")
        print(f"Sample Novelty: {sample_novelty}")
        print(f"Dataset Diversity: {dataset_diversity}, Sample Diversity: {sample_diversity}")

    #load top 100
    top_x = 100
    top_x_smiles = load_top_x_from_full_history(full_history_path, top_x)

    top_x_novelty = novelty(top_x_smiles, dataset_smiles)
    top_x_diversity = diversity(top_x_smiles)
    print(f"Top {top_x} Novelty: {top_x_novelty}, Top {top_x} Diversity: {top_x_diversity}")




