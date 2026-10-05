import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import configparser

import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing

import torch
import tdc
from rdkit import Chem
from torch.utils.data import DataLoader

from dataset import MolTensorDataset
from action_space import Action_Space_GroupSelfies, Action_Space_Smiles
from NMO import Oracle_Handler_GGS, Oracle_Handler_Smiles
from utils import Smiles_MolData
from GGS import GGS
from group_selfies import GroupGrammar
from collections import defaultdict
import numpy as np


def analyze_samples(sequences, action_space, oracle_handler):
    """
    Analyze sampled sequences by converting them to their encodings and evaluating them with the oracle handler.
    Args:
        sequences:
        action_space:
        oracle_handler:

    Returns:

    """
    encodings = []

    for seq in sequences:
        encodings.append(action_space.action_sequence_to_encoding(seq))

    fitness, rewards, oracle_calls_exceeded = oracle_handler.get_fitness(encodings)

    return fitness, rewards

if __name__ == '__main__':


    config_file = sys.argv[1]
    config = configparser.ConfigParser()
    config.read(config_file)

    encoding_type = config.get("General", "encoding_type")
    grammar_path = config.get("General", "grammar_path")
    dataset_path = config.get("Pretrain", "dataset_path")
    batch_size = config.getint("Pretrain", "batch_size")
    log_dir = config.get("Training", "log_dir")

    action_space = None
    data_loader = None

    print(f"Setting up DataLoader for encoding type: {encoding_type}")

    if encoding_type == "GGS":
        grammar = GroupGrammar.from_file(grammar_path)
        action_space = Action_Space_GroupSelfies(grammar)
        oracle_handler = Oracle_Handler_GGS(config_file)
        data = torch.load(dataset_path,
                          map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                          weights_only=False)
        if not isinstance(data, MolTensorDataset):
            raise TypeError("Loaded data is not a MolTensorDataset")

        meta_encoding = data.meta_data_dict.get("encoding_type")
        if meta_encoding != encoding_type:
            raise ValueError(f"Encoding mismatch: config is '{encoding_type}', dataset is '{meta_encoding}'")

        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=True)

    elif encoding_type == "Smiles":
        action_space = Action_Space_Smiles(grammar_path)
        oracle_handler = Oracle_Handler_Smiles(config_file)
        moldata = Smiles_MolData(dataset_path, action_space)
        data_loader = DataLoader(moldata, batch_size=batch_size, shuffle=False, drop_last=True,
                                 collate_fn=Smiles_MolData.collate_fn)
    else:
        raise ValueError(f"Unsupported encoding type: {encoding_type}")

    print(f"DataLoader ready with {len(data_loader)} batches.")

    fitness = []
    sequences = []
    rewards = defaultdict(list)

    for batch in tqdm(data_loader, desc="Calculating Properties"):
        sequences_ = batch[0] if encoding_type == "GGS" else batch
        sequences_ = sequences_.long().cpu().numpy()
        sequences.extend(sequences_)
        fitness_, rewards_ = analyze_samples(sequences_, action_space, oracle_handler)
        fitness += list(fitness_)
        for k, v in rewards_.items():
            rewards[k].extend(v)


    print("\nProcessing complete.")

    # write everything to a file
    output_file = f"{log_dir}/dataset_stats.txt"
    keys_to_skip = ["oracle_calls"]
    keys_to_write = [k for k in rewards.keys() if k not in keys_to_skip]
    print("keys_to_write", keys_to_write)
    lists_to_write = [rewards[k] for k in keys_to_write]

    sequences_str = [' '.join(map(str, seq.tolist())) for seq in sequences]
    header = 'sequence\tfitness\t' + '\t'.join(keys_to_write)
    data_to_save = [sequences_str, fitness] + lists_to_write
    np.savetxt(
        output_file,
        np.column_stack(data_to_save),
        fmt='%s',
        delimiter=',',
        header=header,
        comments=''
    )


    #plotting
    fig, ax = plt.subplots()
    ax.hist(fitness, bins=50, density = True)
    ax.set_xlabel('Fitness')
    ax.set_xlim(0, 1)

    output_file = f"{log_dir}/fitness_dist.svg"
    plt.savefig(output_file)
    plt.close()
    bins = 50
    keys_to_skip = ['oracle_calls_exceeded', 'smiles', 'oracle_calls']
    for k, v in rewards.items():
        print(k)
        if k in keys_to_skip:
            continue
        print("k", k)
        fig, ax = plt.subplots()
        ax.hist(v, bins=bins, density = True)
        ax.set_xlabel(f'{k}')
        ax.set_xlim(0, 1)

        output_file = f"{log_dir}/{k}_dist.svg"
        plt.savefig(output_file)
        plt.close()

        # Save the histogram data
        counts, bin_edges = np.histogram(v, bins=bins, range=(0, 1), density=True)
        hist_data = np.column_stack((bin_edges[:-1], bin_edges[1:], counts))
        output_file_data = f"{log_dir}/{k}_dist_dataset.csv"
        np.savetxt(
            output_file_data,
            hist_data,
            delimiter=",",
            header="bin_start,bin_end,density",
            comments=""
        )





