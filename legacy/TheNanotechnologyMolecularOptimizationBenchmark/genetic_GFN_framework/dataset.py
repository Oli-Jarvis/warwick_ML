import os, sys
import configparser
import random
import torch.utils.data
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from joblib import Parallel, delayed

from utils import padding_and_valid_mask

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create Dataset for GFlow_Mol')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, help='seed', default=-1)
    parser.add_argument('--n_jobs', type=int, help='n_jobs', default=1)
    parser.add_argument(
        "--disable_SMARTS_filters",
        action="store_true",
        help="If set, SMARTS filters are disabled"
    )
    parser.add_argument('config', type=str, help='Path to config file')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    seed = args.seed
    rng = None
    if seed != -1:
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed=seed)
        print("setting seed to ", seed)

from GGS import GGS, disable_rdkit_logging
from group_selfies import GroupGrammar
from action_space import Action_Space_GroupSelfies
from GGS import validate_molecule


class MolTensorDataset(torch.utils.data.TensorDataset):
    def __init__(self, *tensors, meta_data={}):
        super().__init__(*tensors)
        self.meta_data_dict = meta_data


def _generate_batch(seed, batch_size, grammar_path, n_groups, n_explicit_pops,
                    max_seq_length, include_descriptors, use_SMARTS_filters = False):
    """
    Helper function to generate a BATCH of samples in a single worker.
    """

    #setup
    rng = np.random.default_rng(seed)
    ggs = GGS.from_grammar_path(grammar_path=grammar_path, rng=rng)


    if "_RWAS" in grammar_path:
        grammar = GroupGrammar.from_raw_file(grammar_path)
    else:
        grammar = GroupGrammar.from_file(grammar_path)

    action_space = Action_Space_GroupSelfies(grammar)
    end_token_index = action_space.reversed_action_space['End']

    batch_selfies = []
    batch_actions = []
    batch_likelihoods = []
    batch_n_groups = []
    batch_n_atoms = []
    batch_descriptors = []
    invalid_counter = 0

    SMARTS_filters_failed = 0
    counter = 0

    while counter < batch_size:
        n = rng.integers(0, n_groups + 1)
        ggs.create_random_genome(n, n_explicit_pops)
        action_sequence = action_space.encoding_to_action_sequence(ggs.encoding)

        if len(action_sequence) >= max_seq_length:
            invalid_counter += 1
            continue

        n_atoms_ = ggs.mol.GetNumAtoms()
        descriptor_vec = None

        mol = ggs.mol

        if use_SMARTS_filters:
            smiles_ = Chem.MolToSmiles(mol)
            keep, _ = validate_molecule(smiles_)
            if not keep:
                SMARTS_filters_failed += 1
                continue

        if include_descriptors:

            mol.GetRingInfo()
            Chem.SanitizeMol(mol)

            tpsa = Descriptors.TPSA(mol)
            bertzCT = Descriptors.BertzCT(mol)

            num_rings = rdMolDescriptors.CalcNumRings(mol)
            aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
            mol_wt = rdMolDescriptors.CalcExactMolWt(mol)
            hba = rdMolDescriptors.CalcNumLipinskiHBA(mol)
            hbd = rdMolDescriptors.CalcNumLipinskiHBD(mol)
            rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
            fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
            heavy_atoms = rdMolDescriptors.CalcNumHeavyAtoms(mol)
            heteroatoms = rdMolDescriptors.CalcNumHeteroatoms(mol)
            aliph_carbocycles = rdMolDescriptors.CalcNumAliphaticCarbocycles(mol)
            aliph_heterocycles = rdMolDescriptors.CalcNumAliphaticHeterocycles(mol)
            aromatic_carbocycles = rdMolDescriptors.CalcNumAromaticCarbocycles(mol)
            aromatic_heterocycles = rdMolDescriptors.CalcNumAromaticHeterocycles(mol)
            crippen_logp, crippen_mr = rdMolDescriptors.CalcCrippenDescriptors(mol)

            descriptor_vec = [
                tpsa, bertzCT, num_rings, aromatic_rings,
                mol_wt, hba, hbd, rot_bonds, fsp3,
                heavy_atoms, heteroatoms, aliph_carbocycles,
                aliph_heterocycles, aromatic_carbocycles,
                aromatic_heterocycles, crippen_logp, crippen_mr
            ]
            batch_descriptors.append(descriptor_vec)

        action_sequence += [end_token_index]

        batch_selfies.append(ggs.encoding)
        batch_actions.append(action_sequence)
        batch_likelihoods.append(1)
        batch_n_groups.append(len(ggs.groups))
        batch_n_atoms.append(n_atoms_)

        counter += 1

    #print(f"SMARTS filters failed for {SMARTS_filters_failed} molecules in this batch.")

    return (
        batch_selfies,
        batch_actions,
        batch_likelihoods,
        batch_n_groups,
        batch_n_atoms,
        batch_descriptors,
        invalid_counter
    )

def create_pretrain_dataset_group_selfies(filename_out, grammar_path, encoding_type, size, max_seq_length, n_groups, n_explicit_pops,
                                          include_descriptors, use_SMARTS_filters, rng=None, n_jobs = 1):
    """
    Create a dataset for pretraining the RNN model. It can create valid and invalid group selfies
    Args:
        filename_out : str, path to save the dataset
        grammar_path : str, path to the grammar file
        encoding_type : str, type of encoding to use (currently only 'GGS' is supported)
        size : int, total size of the dataset
        max_seq_length : int, maximum length of the sequences
        n_groups : int, maximum number of groups in group selfie
        n_explicit_pops : int, number of explicit pops
        include_descriptors : bool, whether to include descriptors
        use_SMARTS_filters: bool, whether to use SMARTS filters.
        rng (np.random.Generator): random number generator
        n_jobs (int): number of parallel jobs to use
    """

    rng = rng if rng is not None else np.random.default_rng()
    ggs = GGS.from_grammar_path(grammar_path=grammar_path, rng=rng)
    # right now we do not care about duplicates
    if "_RWAS" in grammar_path:
        grammar = GroupGrammar.from_raw_file(grammar_path)
    else:
        grammar = GroupGrammar.from_file(grammar_path)

    action_space = Action_Space_GroupSelfies(grammar)

    valid_group_selfies = []
    valid_action_sequence = []
    valid_likelihood = []
    n_groups_sampled = []
    n_atoms = []
    descriptor_vecs = []
    total_invalid_counter = 0


    actual_n_jobs = n_jobs
    # 3-4 batches per worker.
    n_tasks = actual_n_jobs * 4


    batch_size = int(np.ceil(size / n_tasks))

    print(f"Starting parallel generation...")
    print(f"Total size: {size} samples")
    print(f"CPUs to use: {actual_n_jobs}")
    print(f"Number of parallel tasks: {n_tasks}")
    print(f"Batch size per task: {batch_size}")

    descriptor_names = []
    if include_descriptors:
        descriptor_names = [
            'TPSA', 'BertzCT', 'NumRings', 'NumAromaticRings',
            'MolWt', 'LipinskiHBA', 'LipinskiHBD', 'NumRotatableBonds', 'FractionCSP3',
            'NumHeavyAtoms', 'NumHeteroatoms', 'NumAliphaticCarbocycles',
            'NumAliphaticHeterocycles', 'NumAromaticCarbocycles',
            'NumAromaticHeterocycles', 'CrippenLogP', 'CrippenMR'
        ]

    #Generate seeds for each worker
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)  # Handle int seed

    child_seeds = rng.integers(0, 2 ** 32 - 1, size=n_tasks)

    tasks = [
        delayed(_generate_batch)(
            seed=child_seeds[i],
            batch_size=batch_size,
            grammar_path=grammar_path,
            n_groups=n_groups,
            n_explicit_pops=n_explicit_pops,
            max_seq_length=max_seq_length,
            include_descriptors=include_descriptors,
            use_SMARTS_filters=use_SMARTS_filters
        )
        for i in range(n_tasks)
    ]

    results = Parallel(n_jobs=n_jobs, verbose=10)(tasks)


    print("Processing results...")
    for res in results:

        (b_selfie, b_act, b_like, b_grp, b_atom, b_desc, b_invalid) = res

        valid_group_selfies.extend(b_selfie)
        valid_action_sequence.extend(b_act)
        valid_likelihood.extend(b_like)
        n_groups_sampled.extend(b_grp)
        n_atoms.extend(b_atom)
        descriptor_vecs.extend(b_desc)
        total_invalid_counter += b_invalid

    valid_group_selfies = valid_group_selfies[:size]
    valid_action_sequence = valid_action_sequence[:size]
    valid_likelihood = valid_likelihood[:size]
    n_groups_sampled = n_groups_sampled[:size]
    n_atoms = n_atoms[:size]
    descriptor_vecs = descriptor_vecs[:size]

    print(f"Number of valid sequences: {len(valid_action_sequence)}")

    padded_sequences = torch.zeros(len(valid_action_sequence), max_seq_length)
    valid_mask = torch.zeros(len(valid_action_sequence), max_seq_length, dtype=torch.bool)
    # set all values to end token index
    end_token_index = action_space.reversed_action_space['End']
    padded_sequences.fill_(end_token_index)
    for i, seq in enumerate(valid_action_sequence):
        padded_sequences[i, :len(seq)] = torch.tensor(seq)
        valid_mask[i, :len(seq) - 1] = 1

    # action_sequences = torch.tensor(action_sequences, dtype=torch.float32)
    likelihoods = torch.tensor(valid_likelihood, dtype=torch.float32)

    # set up dataset for pytorch
    meta_data = {"grammar_path": grammar_path, "max_seq_length": max_seq_length, "n_explicit_pops": n_explicit_pops, "encoding_type": encoding_type}
    if include_descriptors:
        descriptor_vecs_tensor = torch.tensor(descriptor_vecs, dtype=torch.float32)

        meta_data["descriptor_names"] = descriptor_names


        dataset = MolTensorDataset(padded_sequences, valid_mask, descriptor_vecs_tensor, meta_data=meta_data)
    else:
        dataset = MolTensorDataset(padded_sequences, valid_mask, meta_data=meta_data)

    # save dataset to file
    torch.save(dataset, filename_out)

    # plot histogram of n_atoms
    plt.hist(n_atoms, bins=range(1, max(n_atoms) + 2), align='left', rwidth=0.8)
    plt.xlabel('Number of Atoms')
    plt.ylabel('Frequency')
    path = filename_out.rsplit('.', 1)[0]
    plt.savefig(f"{path}_histogram_atoms.svg")
    plt.clf()

    # plot histogram of n_groups
    plt.hist(n_groups_sampled, bins=range(1, max(n_groups_sampled) + 2), align='left', rwidth=0.8)
    plt.xlabel('Number of Groups')
    plt.ylabel('Frequency')
    path = filename_out.rsplit('.', 1)[0]
    plt.savefig(f"{path}_histogram_groups.svg")

    if include_descriptors:
        descriptor_vecs_array = np.array(descriptor_vecs)
        for i in range(descriptor_vecs_array.shape[1]):
            plt.clf()
            plt.hist(descriptor_vecs_array[:, i], bins=50, rwidth=0.8)
            plt.xlabel(descriptor_names[i])
            plt.ylabel('Frequency')
            plt.savefig(f"{path}_histogram_{descriptor_names[i]}.svg")


if __name__ == '__main__':
    disable_rdkit_logging()
    config_file = args.config
    config = configparser.ConfigParser()
    config.read(config_file, encoding='utf-8')

    encoding_type = config.get('General', 'encoding_type')
    max_seq_length = config.getint('General', 'max_seq_length')
    grammar_path = config.get('General', 'grammar_path')

    out_path = config.get('Dataset', 'dataset_path')
    size = config.getint('Dataset', 'size')
    n_groups = config.getint('Dataset', 'n_groups')
    n_explicit_pops = config.getint('Dataset', 'n_explicit_pops')
    include_descriptors = config.getboolean('Dataset', 'include_descriptors', fallback=False)
    use_SMARTS_filters = config.getboolean('Dataset', 'use_SMARTS_filters', fallback=False)

    #print all option read from config
    print("--------- Configuration: ----------")
    for section in config.sections():
        for key, value in config.items(section):
            print(f"{section}.{key} = {value}")
    print("-----------------------------------")

    create_pretrain_dataset_group_selfies(out_path, grammar_path, encoding_type, size, max_seq_length, n_groups, n_explicit_pops, include_descriptors, use_SMARTS_filters,
                                          rng=rng, n_jobs = args.n_jobs)