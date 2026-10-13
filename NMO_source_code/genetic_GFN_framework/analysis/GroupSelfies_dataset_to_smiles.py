import concurrent
import sys
import os
from functools import partial

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


def translate_samples(sequences, action_space, oracle_handler):
    """
    translate sampled sequences by converting them to their encodings and returning smiles
    Args:
        sequences:
        action_space:
        oracle_handler:

    Returns:

    """
    smiles = []

    for seq in sequences:
        seq = seq.tolist() if isinstance(seq, torch.Tensor) else seq
        encoding = action_space.action_sequence_to_encoding(seq)
        ggs = GGS.from_grammar_path(grammar_path=oracle_handler.grammar_path, encoding=encoding)
        mol = ggs.mol
        Chem.SanitizeMol(mol)
        smiles_ = Chem.MolToSmiles(ggs.mol)
        smiles.append(smiles_)

    return smiles


def process_batch(batch, encoding_type, action_space, oracle_handler):
    """
    Encapsulates the logic for processing a single batch.
    This function will be executed in a separate process.
    """
    sequences = batch[0] if encoding_type == "GGS" else batch

    return translate_samples(sequences, action_space, oracle_handler)

if __name__ == '__main__':

    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass


    config_file = sys.argv[1]
    config = configparser.ConfigParser()
    config.read(config_file)

    encoding_type = config.get("General", "encoding_type")
    grammar_path = config.get("General", "grammar_path")
    dataset_path = config.get("Dataset", "dataset_path")
    batch_size = config.getint("Pretrain", "batch_size")
    pretrain_output_path = config.get("Pretrain", "output_path")
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

    else:
        raise ValueError(f"Unsupported encoding type: {encoding_type}")

    print(f"DataLoader ready with {len(data_loader)} batches.")

    worker_func = partial(process_batch,
                          encoding_type=encoding_type,
                          action_space=action_space,
                          oracle_handler=oracle_handler)

    smiles = []
    max_workers = 16
    with concurrent.futures.ProcessPoolExecutor(max_workers= max_workers) as executor:

        futures = []
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                cpu_batch = [t.cpu() for t in batch]
            else:
                cpu_batch = batch.cpu()
            futures.append(executor.submit(worker_func, cpu_batch))

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Translating to Smiles"):
            try:
                smiles_batch = future.result()
                smiles.extend(smiles_batch)
            except Exception as e:
                print(f"\nA worker process failed with an error: {e}")

    #write smiles to file
    print("writing smiles to file")
    with open(f"{pretrain_output_path}/translated_smiles.smi", "w") as f:
        for smi in smiles:
            f.write(smi + "\n")





