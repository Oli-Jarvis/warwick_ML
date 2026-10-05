import sys
import os
import random
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np
from tqdm import tqdm
from collections import defaultdict
import torch
import matplotlib.pyplot as plt
import configparser
from rdkit import Chem
from group_selfies import GroupGrammar
import pandas as pd

from model import get_model
from action_space import Action_Space_GroupSelfies, Action_Space_Smiles
from NMO import Oracle_Handler_GGS, Oracle_Handler_Smiles
from train_utils import CheckpointHandler
from utils import unique, Experience, Experience_Item
from GGS import validate_molecule

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NEW_CPUS = 12

def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    import rdkit.rdBase as rkrb
    import rdkit.RDLogger as rkl
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog('rdApp.error')


def load_history_to_cache(file_path):
    cache = {}
    if os.path.exists(file_path):
        try:
            # Read history and drop duplicates based on the encoding blueprint
            df = pd.read_csv(file_path, sep=';')
            df = df.drop_duplicates(subset=['encoding'], keep='first')

            # Identify property columns (SA, hl_gaps, k_ph, etc.)
            exclude = ['action_sequence', 'encoding', 'fitness',
                       'oracle_calls', 'mutation_stats', 'crossover_stats', 'created_by', 'step', 'generation']
            reward_cols = [c for c in df.columns if c not in exclude]

            for _, row in df.iterrows():
                enc = row['encoding']
                if pd.isna(enc): continue

                # Store the fitness and the dict of specific rewards
                fitness = row['fitness']
                rewards = {k: row[k] for k in reward_cols}
                cache[enc] = (fitness, rewards)

            print(f"Loaded {len(cache)} unique encodings from {file_path}")
        except Exception as e:
            print(f"Error loading history: {e}")
    return cache

def sample_agent(encoding_type, agent, oracle_handler, batch_size, n_samples, history_path = None, grammar = None):
    """
    Sample sequences from the agent and evaluate them using the oracle handler.
    Args:
        encoding_type (String): "Smiles" or "GGS"
        agent:
        oracle_handler:
        batch_size:
        n_samples:

    Returns:

    """

    newly_discovered = Experience(agent.action_space, max_size=-1, strict=True)
    if history_path:
        encoding_cache = load_history_to_cache(history_path)
    else:
        encoding_cache = {}
    smiles_cache = {}



    n_steps = n_samples // batch_size + 1
    fitness_all = []
    sequences_all = []
    rewards_all = defaultdict(list)
    duplicates = 0
    invalid = 0
    filtered = 0
    new_unique_candiates = 0

    for step in tqdm(range(n_steps)):
        encodings_step = []
        sequences_, valid_mask, agent_likelihood, entropy = agent.sample_sequences(batch_size)
        sequences_ = sequences_.detach()

        #count duplicates
        inital_n_sample = sequences_.shape[0]
        unique_idxs = unique(sequences_)
        duplicates += inital_n_sample - len(unique_idxs)

        if torch.cuda.is_available():
            sequences_ = sequences_.cpu().numpy()
        else:
            sequences_ = sequences_.numpy()

        print(f"sampled_sequences_ in step {step}: {sequences_}")

        sequences_to_keep = []
        encodings_to_keep = []
        candidates = []
        for i, sequence in enumerate(sequences_):
            encoding = agent.action_space.action_sequence_to_encoding(sequence)
            encodings_step.append(encoding)


            if not encoding or encoding == "":
                invalid += 1
                continue

            mol = None
            smiles_canonical = None
            keep = False
            reason = []

            if encoding_type == "Smiles":
                keep, reason = validate_molecule(encoding)
                if keep:
                    mol = Chem.MolFromSmiles(encoding)
                    if mol:
                        smiles_canonical = Chem.MolToSmiles(mol)
                    else:
                        keep = False
            elif encoding_type == "GGS":
                try:
                    mol_decoded = grammar.decoder(encoding)
                    smiles_canonical = Chem.MolToSmiles(mol_decoded)
                    keep, reason = validate_molecule(smiles_canonical)
                    mol = mol_decoded
                except ValueError:
                    keep = False
                    reason = ["Invalid SMILES"]
            if keep:
                encodings_to_keep.append(encoding)
                sequences_to_keep.append(sequence)
                n_atoms = mol.GetNumAtoms()
                candidates.append((i, sequence, encoding, smiles_canonical, n_atoms))
            else:
                if "Invalid SMILES" in reason:
                    invalid += 1
                else:
                    filtered += 1

        #print(f"candidates in step {step}: {candidates}")

        indices_to_compute = []
        encodings_to_compute = []
        smis_to_compute = []
        batch_results = [None] * len(candidates)

        for idx, (original_idx, seq, enc, smi, n_atoms) in enumerate(candidates):
            #data from history cache
            if enc in encoding_cache:
                batch_results[idx] = encoding_cache[enc]
                #print(f"found enc in cache: {enc}")
            #data from sample cache
            elif smi in smiles_cache:
                batch_results[idx] = smiles_cache[smi]
                #print(f"found smi in cache: {smi}")
            else:
                #check duplicates within batch
                if smi not in smis_to_compute and smi not in smiles_cache:
                    indices_to_compute.append(idx)
                    encodings_to_compute.append(enc)
                    smis_to_compute.append(smi)
                    new_unique_candiates += 1


        if encodings_to_compute:
            new_fitness, new_rewards, _ = oracle_handler.get_fitness(encodings_to_compute)

            # Update cache and history with unique results
            for j, comp_idx in enumerate(indices_to_compute):
                fit_val = new_fitness[j]
                rew_val = {k: v[j] for k, v in new_rewards.items()}
                _, seq, enc, smi, n_atoms = candidates[comp_idx]

                smiles_cache[smi] = (fit_val, rew_val)

                new_experience = Experience_Item(
                    action_sequence=seq, encoding=enc, fitness=fit_val,
                    reward=rew_val, n_atoms=n_atoms, meta_data={}, strict=True
                )
                print(new_experience)
                newly_discovered.add_experience(new_experience)

        # final pass
        for idx, (_, _, _, smi, _) in enumerate(candidates):
            if batch_results[idx] is None:
                batch_results[idx] = smiles_cache[smi]

        # aggregate results
        for res, (original_idx, seq, _, _, _) in zip(batch_results, candidates):
            if res is None: continue
            fit_val, rew_val = res
            sequences_all.append(seq)
            fitness_all.append(fit_val)
            for k, v in rew_val.items():
                rewards_all[k].append(v)

    duplicate_rate = duplicates / n_samples
    invalid_rate = invalid / n_samples
    filtered_rate = filtered / n_samples
    new_unique_rate = new_unique_candiates / n_samples

    return sequences_all, fitness_all, rewards_all, duplicate_rate, invalid_rate, filtered_rate, new_unique_rate, newly_discovered

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
    import argparse
    parser = argparse.ArgumentParser(description='Train GFlow_Mol')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, help='seed', default = -1)
    parser.add_argument("config", type=str, help='Path to config file')
    parser.add_argument("n_samples", type=int, help='How many samples to draw')
    parser.add_argument("agent", type=str, help='Which agent to sample from, final or prior')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    seed = args.seed
    if seed != -1:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed=seed)

    config_file = args.config
    n_samples = args.n_samples
    agent = args.agent

    disable_rdkit_logging()

    config = configparser.ConfigParser()
    config.read(config_file, encoding='utf-8')
    log_dir = config.get("Training", "log_dir")
    learning_rate = config.getfloat("Training", "learning_rate")
    learning_rate_z = config.getfloat("Training", "learning_rate_z")
    batch_size = config.getint("Training", "batch_size")
    grammar_path = config.get("General", "grammar_path")
    max_seq_length = config.getint("General", "max_seq_length")
    encoding_type = config.get("General", "encoding_type")
    prior_path = config.get("Training", "prior_path")
    num_layers = config.getint("General", "num_layers")
    d_model = config.getint("General", "d_model")
    model = config.get("General", "model")

    original_log_dir = config.get("Training", "log_dir")
    new_log_dir = os.path.join(original_log_dir, "sampling")
    os.makedirs(new_log_dir, exist_ok=True)
    config.set("Training", "log_dir", new_log_dir)
    config.set("Oracle", "n_cpus_total", str(NEW_CPUS))
    new_config_path = os.path.join(new_log_dir, "sampling_config.ini")
    with open(new_config_path, 'w', encoding='utf-8') as f:
        config.write(f)

    if encoding_type == "GGS":
        action_space = Action_Space_GroupSelfies.from_grammar_path(grammar_path)
        oracle_handler = Oracle_Handler_GGS(new_config_path)
        grammar = GroupGrammar.from_file(grammar_path)
    elif encoding_type == "Smiles":
        action_space = Action_Space_Smiles(grammar_path)
        oracle_handler = Oracle_Handler_Smiles(new_config_path)
        grammar = None

    Agent = get_model(model, action_space, batch_size, max_seq_length, num_layers=num_layers, d_model=d_model)
    Agent.to(device)
    if agent == "final":
        agent_path = f"{log_dir}/agent_final.pt"
    elif agent == "prior":
        agent_path = f"{prior_path}"
    else:
        raise ValueError(f"Unknown agent: {agent}")


    if torch.cuda.is_available():
        state_dict = torch.load(agent_path, weights_only=False)
    else:
        state_dict = torch.load(agent_path, map_location=torch.device('cpu'), weights_only=False)

    # register buffers
    if 'desc_mean' in state_dict:
        Agent.net.register_buffer('desc_mean', state_dict['desc_mean'])
        Agent.net.register_buffer('desc_std', state_dict['desc_std'])

        print("Registered descriptor statistics to Agent.")

    Agent.net.load_state_dict(state_dict, strict=False)


    desc_mean = getattr(Agent.net, 'desc_mean', None)
    desc_std = getattr(Agent.net, 'desc_std', None)

    if desc_mean is not None:
        print(f"Loaded descriptor statistics: Mean={desc_mean.mean().item():.2f}, Std={desc_std.mean().item():.2f}")
    else:
        print("Warning: No descriptor statistics found in model. Descriptor loss will be skipped.")

    # set up partition function Z -> here log_z
    if torch.cuda.is_available():
        log_z = torch.nn.Parameter(torch.tensor([5.], device = 'cuda:0'))
    else:
        log_z = torch.nn.Parameter(torch.tensor([5.]))

    optimizer = torch.optim.Adam([{'params': Agent.net.parameters(), 'lr': learning_rate},
                                      {'params' : log_z, 'lr' : learning_rate_z}])

    ckpt_handler = CheckpointHandler(log_dir)
    #load
    start_step, loaded_exp, loaded_hist = ckpt_handler.attempt_restart(
        Agent, optimizer, log_z, oracle_handler, prior_path, device, rng
    )
    if loaded_exp: experience = loaded_exp
    if loaded_hist: full_history = loaded_hist


    print("Switching model to evaluation mode...")
    Agent.net.eval()

    sequences, fitness, rewards, duplicate_rate, invalid_rate, filtered_rate, new_unique_rate, newly_discovered  = sample_agent(encoding_type, Agent,
                                                                                             oracle_handler,
                                                                                             batch_size,
                                                                                             n_samples=n_samples,
                                                                                             history_path = f"{original_log_dir}/full_history.csv",
                                                                                             grammar = grammar)

    #plotting
    fig, ax = plt.subplots()
    ax.hist(fitness, bins=100, density = True)
    ax.set_xlabel('Fitness')
    ax.set_xlim(0, 1)

    output_file = f"{new_log_dir}/fitness_dist_{agent}.svg"
    plt.savefig(output_file)
    plt.close()

    keys_to_skip = ['oracle_calls_exceeded', 'smiles', 'oracle_calls']
    for k, v in rewards.items():
        print(k)
        if k in keys_to_skip:
            continue
        print("k", k)
        fig, ax = plt.subplots()
        ax.hist(v, bins=50, density = True)
        ax.set_xlabel(f'{k}')
        ax.set_xlim(0, 1)

        output_file = f"{new_log_dir}/{k}_dist_{agent}.svg"
        plt.savefig(output_file)
        plt.close()

    # write everything to a file
    output_file = f"{new_log_dir}/sampled_sequences_{agent}_stats.csv"

    keys_to_skip = ['oracle_calls_exceeded', 'oracle_calls']
    keys_to_write = [k for k in rewards.keys() if k not in keys_to_skip]
    lists_to_write = [rewards[k] for k in keys_to_write]

    sequences_str = [' '.join(map(str, seq.tolist())) for seq in sequences]
    header = 'sequence;fitness;' + ';'.join(keys_to_write)
    data_to_save = [sequences_str, fitness] + lists_to_write
    np.savetxt(
        output_file,
        np.column_stack(data_to_save),
        fmt='%s',
        delimiter=';',
        header=header,
        comments=''
    )

    newly_discovered.write_memory(f"{new_log_dir}/newly_discovered_{agent}.csv")

    with open(f"{new_log_dir}/sampled_sequences_{agent}_summary.txt", 'w') as f:
        f.write(f"Total samples: {n_samples}\n")
        f.write(f"Duplicate rate: {duplicate_rate:.4f}\n")
        f.write(f"Invalid rate: {invalid_rate:.4f}\n")
        f.write(f"Filtered rate: {filtered_rate / n_samples:.4f}\n")




