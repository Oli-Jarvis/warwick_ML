from dataclasses import dataclass
from typing import List, Dict, Any

import numpy
import pandas as pd
from group_selfies import GroupGrammar
import numpy as np
import torch
from sympy.physics.units import length
from torch.utils.data import Dataset
import rdkit.Chem as Chem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors


@dataclass
class Experience_Item(object):
    action_sequence: List[int]
    encoding: str
    fitness: float
    reward: Dict[str, Any]
    n_atoms: int
    meta_data: Dict[str, Any] = None
    #If equal is strict -> n_atoms and fitness must match -> can be used to avoid duplicates in experience replay
    strict: bool = True


    def __eq__(self, other):
        if not isinstance(other, Experience_Item):
            return False
        

        # for smiles encoding, smiles is not an extra key in reward dict
        # todo: improve performance here
        #try:
        #    me_mol = Chem.MolFromSmiles(self.reward["smiles"])
        #    other_mol = Chem.MolFromSmiles(other.reward["smiles"])
        #except KeyError:
        #    me_mol = self.encoding
        #    other_mol = other.encoding
        #    me_mol = Chem.MolFromSmiles(me_mol)
        #    other_mol = Chem.MolFromSmiles(other_mol)
        #    if not other_mol:
        #        return False
        #    if not me_mol:
        #        return False


        #n_me = me_mol.GetNumAtoms()
        #n_other = other_mol.GetNumAtoms()

        smiles_match = False
        if "smiles" in self.reward and "smiles" in other.reward:
            smiles_match = (self.reward["smiles"] == other.reward["smiles"])

        if self.strict:
            encoding_match = (self.encoding == other.encoding)
            match = (self.n_atoms == other.n_atoms and self.fitness == other.fitness) or encoding_match
        else:
            action_sequences_match = np.array_equal(np.asarray(self.action_sequence), np.asarray(other.action_sequence))
            match = action_sequences_match

        return match

class Experience(object):

    def __init__(self, action_space, max_size = 100, strict = True, seed = -1):
        self.action_space = action_space
        self.max_size = max_size
        self.memory: List[Experience_Item] = []
        self.strict = strict
        self.seed = seed
        if self.seed  != -1:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)


    def add_experience_batch(self, action_sequences, encodings, fitness, rewards_dict, meta_data=None):
        """
        Adds experience to memory. It is checked if the experience is already in memory. By default, the memory is sorted by reward.
        The maximum size of the memory is set to 100. If the memory is full, the experience with lowest reward is removed.
        Args:
            action_sequences : list of action sequences
            encodings : list of encodings
            fitness: list of fitness values
            rewards_dict : dict of rewards, where keys are reward types and values are lists of rewards
            meta_data: dict or list of dicts -> usually a batch has the same meta data, so it can be a single dict
        """
        if (len(action_sequences) != len(encodings) or len(encodings) != len(fitness)):
            raise ValueError(f"The length of action_sequences and group_selfies and rewards must match: {len(action_sequences)=}, {len(encodings)=},{len(fitness)=}, {len(rewards_dict[list(rewards_dict.keys())[0]])=}")

        for i in range(len(action_sequences)):
            #check if meta_data is a single dict or a list/array of dicts
            if meta_data and isinstance(meta_data, dict):
                meta = meta_data
            elif meta_data and isinstance(meta_data, list) or isinstance(meta_data, np.ndarray):
                meta = meta_data[i] if i < len(meta_data) else {}
            else:
                meta = {}
            rewards = [{key: rewards_dict[key][i] for key in rewards_dict} for i in range(len(action_sequences))]
            smiles = rewards[i]["smiles"]
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                n_atoms = mol.GetNumAtoms()
            else:
                n_atoms = 0
            encoding_str = str(encodings[i])
            experience = Experience_Item(action_sequences[i], encoding_str, fitness[i], rewards[i], n_atoms, meta_data=meta, strict = self.strict)
            self.add_experience(experience)

    def add_incomplete_experience_batch(self, encodings, fitness, rewards_dict, meta_data = None):
        """
        Works the same as add_experience_batch, but only adds the experience if encoding is valid -> might be not valid if
        genetic operations are applied on encoding level. It´s called incomplete because the action sequence is not provided
        but generated from the encoding.
        Args:
            encodings : list of encodings
            fitness: list of fitness values
            rewards : dict of rewards
            meta_data: dict or list of dicts -> usually a batch has the same meta data, so it can be a single dict
        """
        if len(fitness) != len(encodings):
            raise ValueError("The length of encodings and rewards must match")
        for i in range(len(encodings)):
            encoding_str = str(encodings[i])
            try:
                action_sequence = np.array(self.action_space.encoding_to_action_sequence(encodings[i]))

                if meta_data and isinstance(meta_data, dict):
                    meta = meta_data
                elif meta_data and isinstance(meta_data, list) or isinstance(meta_data, np.ndarray):
                    meta = meta_data[i] if i < len(meta_data) else {}
                else:
                    meta = {}

                if "use_max_seq_length_padding" in meta.keys() and meta["use_max_seq_length_padding"]:
                    action_sequence = padding_and_valid_mask(sequence[i])
            except KeyError as e:
                print(f"Encoding {encodings[i]} is not valid: {e}")
                action_sequence = []
                meta = {}

            reward_dict_list = [{key: rewards_dict[key][i] for key in rewards_dict} for i in range(len(encodings))]
            smiles = reward_dict_list[i]["smiles"]
            mol = Chem.MolFromSmiles(smiles)
            n_atoms = mol.GetNumAtoms()
            experience = Experience_Item(action_sequence, encoding_str, fitness[i], reward_dict_list[i], n_atoms, meta_data=meta, strict = self.strict)
            self.add_experience(experience)

    def add_experience(self, experience: Experience_Item):
        """
        Adds experience to memory. It is checked if the experience is already in memory and if experience item has an end_token.
        By default, the memory is sorted by reward. The maximum size of the memory is set to 100. If the memory is full,
        the experience with lowest reward is removed. The memory is always padded to the maximum size
        Args:
            experience : Experience_Item object
        """

        has_end_token = self.action_space.has_end_token(experience.action_sequence)

        #see if experience is already in memory -> skip for full history where max_size = -1
        if self.max_size != -1 and experience in self.memory:
            #print(f"Experience {experience} already in memory")
            pass
        elif self.max_size != -1 and not has_end_token:
            pass
        else:
            self.memory.append(experience)
            self.memory = sorted(self.memory, key=lambda m: m.fitness, reverse=True)

            #if memory is full, remove the oldest experience and lowest reward
            if self.max_size != -1 and len(self.memory) > self.max_size:
                self.memory.pop()

    def sequence_padding(self, action_sequences):
        """
        Pad action sequences to the maximum length in the batch.
        Args:
            action_sequences : list of action sequences, each sequence is a list of integers
        Returns:
            padded_sequences : np.array, padded action sequences with shape (batch_size, max_length)
        """
        max_length = max([len(seq) for seq in action_sequences])
        end_token_index = self.action_space.reversed_action_space['End']
        padded_sequences = np.full((len(action_sequences), max_length), end_token_index, dtype=int)

        for i, seq in enumerate(action_sequences):
            padded_sequences[i, :len(seq)] = torch.tensor(seq)

        return padded_sequences

    def rank_based_sample(self, n, rank_coefficient=0.01, return_memory_item=False, return_smiles=False):
        """
        Sample n experiences from memory based on their rank.
        Items with zero fitness are excluded.
        If fewer than n items have non-zero fitness, returns all available valid items (sampled).
        Args:
            return_smiles: If True, returns a list of SMILES strings alongside sequences and fitness.
        """
        all_fitness = np.array([exp.fitness for exp in self.memory])

        # Filter zero fitness
        valid_mask = ~np.isclose(all_fitness, 0.0)
        valid_indices = np.where(valid_mask)[0]
        num_valid = len(valid_indices)

        if num_valid == 0:
            if return_memory_item:
                return []
            elif return_smiles:
                return [], [], []
            else:
                return [], []

        current_n = min(n, num_valid)
        valid_fitness = all_fitness[valid_indices]

        # calculate ranks
        ranks = np.argsort(np.argsort(-1 * valid_fitness))
        weights = 1.0 / (rank_coefficient * num_valid + ranks)

        if self.seed == -1:
            generator = None
        else:
            generator = self.generator

        relative_indices = list(torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=current_n,
            replacement=True,
            generator=generator
        ))

        final_indices = valid_indices[relative_indices]

        if return_memory_item:
            return [self.memory[i] for i in final_indices]
        else:
            sampled_sequences = [self.memory[i].action_sequence for i in final_indices]
            sampled_sequences = self.sequence_padding(sampled_sequences)
            sampled_fitness = [self.memory[i].fitness for i in final_indices]

            if return_smiles:
                # Retrieve SMILES from the reward dictionary
                sampled_smiles = [self.memory[i].reward.get('smiles') for i in final_indices]
                return sampled_sequences, sampled_fitness, sampled_smiles

            return sampled_sequences, sampled_fitness

    def uniform_sample(self, n, return_memory_item=False, return_smiles=False):
        """
        Sample n experiences from memory UNIFORMLY.
        Implemented using WeightedRandomSampler with equal weights to ensure
        consistency with the seeding mechanisms of rank_based_sample.
        """
        # valid items
        all_fitness = np.array([exp.fitness for exp in self.memory])
        valid_mask = ~np.isclose(all_fitness, 0.0)
        valid_indices = np.where(valid_mask)[0]
        num_valid = len(valid_indices)

        # handle empty mem
        if num_valid == 0:
            if return_memory_item: return []
            elif return_smiles: return [], [], []
            else: return [], []

        current_n = min(n, num_valid)

        weights = torch.ones(num_valid, dtype=torch.double)

        if self.seed == -1:
            generator = None
        else:
            generator = self.generator

        relative_indices = list(torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=current_n,
            replacement=True,
            generator=generator
        ))

        final_indices = valid_indices[relative_indices]

        if return_memory_item:
            return [self.memory[i] for i in final_indices]
        else:
            sampled_sequences = [self.memory[i].action_sequence for i in final_indices]
            sampled_sequences = self.sequence_padding(sampled_sequences)
            sampled_fitness = [self.memory[i].fitness for i in final_indices]

            if return_smiles:
                sampled_smiles = [self.memory[i].reward.get('smiles') for i in final_indices]
                return sampled_sequences, sampled_fitness, sampled_smiles

            return sampled_sequences, sampled_fitness


    def write_memory(self, filepath, sort_by = None):
        """
        Write the memory to csv file using pandas.
        Args:
            filepath : str, path to the file
        """

        list_of_rows = []
        for experience in self.memory:
            #if value of a key is array-like, it should not be written to the csv file
            #filtered_reward = {
            #    k: v for k, v in experience.reward.items()
            #    if not isinstance(v, (list, tuple, np.ndarray))
            #}
            row = {**{'action_sequence': str(experience.action_sequence), 'encoding': str(experience.encoding), 'fitness': experience.fitness},
                   **experience.reward, **experience.meta_data}
            list_of_rows.append(row)
        dataframe = pd.DataFrame(list_of_rows)
        if sort_by:
            dataframe = dataframe.sort_values(sort_by, ascending=False)
        #see if duplicate values for "oracle_calls" exist
        duplicate = dataframe.duplicated(subset=["oracle_calls"]).any()
        assert duplicate == False, "Duplicate values for oracle_calls exist in memory, this should not happen"
        dataframe = dataframe.drop_duplicates(subset=["oracle_calls"])
        dataframe = dataframe.reset_index(drop=True)
        for col in dataframe.select_dtypes(include=['object']).columns:
            dataframe[col] = dataframe[col].str.replace(r'[\r\n]+', ' ', regex=True)
        dataframe.to_csv(filepath, index=False, sep=";")

    def encodings_in_memory(self, encodings_to_check: List[str]):
        """
        Checks if the encodings are already in memory and returns array of boolean values to indicate indices of
        encodings that are in memory.
        """
        existing_encodings_set = {str(item.encoding) for item in self.memory}
        #print("existing_encodings_set")
        #print(existing_encodings_set)
        #print("-------------------------------------")
        #print("encodings_to_check")
        #print(encodings_to_check)
        #print("-------------------------------------")
        match_array = np.array([str(encoding) in existing_encodings_set for encoding in encodings_to_check], dtype=bool)
        #print(match_array)
        #print("-------------------------------------")
        return match_array

    def get_max_oracle_calls(self):
        """
        Gives the max oracle calls in memory
        Returns:

        """
        list_ = [item.reward['oracle_calls'] for item in self.memory]
        if len(list_) == 0:
            return 0
        max_oracle_calls = max(list_)
        return max_oracle_calls

    def get_lifetime_best_candidate(self, max_oracle_calls = -1):
        """
        Gives the best candidate in memory based on fitness
        Returns:

        """
        if max_oracle_calls == -1:
            max_oracle_calls = self.get_max_oracle_calls()
        if(len(self.memory) == 0):
            return 0
        best_item = max(self.memory, key=lambda item: item.fitness)
        oracle_calls_best_item = best_item.reward['oracle_calls']
        lifetime = max_oracle_calls - oracle_calls_best_item
        return lifetime

def calc_descriptors(mol):
    """
    Calculate descriptors for a molecule, matching the logic in dataset.py.
    Returns a list of 18 float values.
    """
    try:
        Chem.SanitizeMol(mol)
        mol.GetRingInfo()

        # 1. Simple Descriptors
        tpsa = Descriptors.TPSA(mol)
        bertzCT = Descriptors.BertzCT(mol)

        # 2. Molecular Graph Descriptors
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        mol_wt = rdMolDescriptors.CalcExactMolWt(mol)
        hba = rdMolDescriptors.CalcNumLipinskiHBA(mol)
        hbd = rdMolDescriptors.CalcNumLipinskiHBD(mol)
        rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
        heavy_atoms = rdMolDescriptors.CalcNumHeavyAtoms(mol)
        heteroatoms = rdMolDescriptors.CalcNumHeteroatoms(mol)

        # 3. Cycle counts
        aliph_carbocycles = rdMolDescriptors.CalcNumAliphaticCarbocycles(mol)
        aliph_heterocycles = rdMolDescriptors.CalcNumAliphaticHeterocycles(mol)
        aromatic_carbocycles = rdMolDescriptors.CalcNumAromaticCarbocycles(mol)
        aromatic_heterocycles = rdMolDescriptors.CalcNumAromaticHeterocycles(mol)

        # 4. Crippen
        crippen_logp, crippen_mr = rdMolDescriptors.CalcCrippenDescriptors(mol)

        descriptor_vec = [
            tpsa, bertzCT, num_rings, aromatic_rings,
            mol_wt, hba, hbd, rot_bonds, fsp3,
            heavy_atoms, heteroatoms, aliph_carbocycles,
            aliph_heterocycles, aromatic_carbocycles,
            aromatic_heterocycles, crippen_logp, crippen_mr
        ]
        return descriptor_vec

    except Exception as e:
        # If descriptor calculation fails for any reason (e.g. sanitization error), return None
        return None




class Smiles_MolData(Dataset):
    """Custom PyTorch Dataset that takes a file containing SMILES.
    Taken from https://github.com/hyeonahkimm/genetic_gfn/blob/main/sars_cov2/genetic_gfn/data_structs.py

        Args:
                fname : path to a file containing \n separated SMILES.
                voc   : a Vocabulary instance

        Returns:
                A custom PyTorch dataset for training the Prior.
    """
    def __init__(self, fname, voc):
        self.voc = voc
        self.smiles = []
        with open(fname, 'r') as f:
            for line in f:
                self.smiles.append(line.split()[0])

    def __getitem__(self, i):
        encoded = self.voc.encoding_to_action_sequence(self.smiles[i])
        encoded = torch.tensor(encoded).long()
        return encoded

    def __len__(self):
        return len(self.smiles)

    def __str__(self):
        return "Dataset containing {} structures.".format(len(self))

    @classmethod
    def collate_fn(cls, arr, fill_value):
        """Function to take a list of encoded sequences and turn them into a batch"""
        max_length = max([seq.size(0) for seq in arr])

        array = torch.full((len(arr), max_length), fill_value=fill_value)

        if torch.cuda.is_available():
            collated_arr = array.cuda()
        else:
            collated_arr = array

        for i, seq in enumerate(arr):
            collated_arr[i, :seq.size(0)] = seq
        return collated_arr




def unique(arr):
    """
    Returns the unique indices of a 2D numpy array or torch tensor.
    If the input is a torch tensor, a tensor of unique indices is returned.
    If the input is a numpy array, a numpy array of unique indices is returned.
    Args:
        arr : numpy array or torch tensor of shape (n, m) where n is the number of samples and m is the number of features
    Returns:
        unique_indices : numpy array or torch tensor of unique indices of shape (k,) where k is the number of unique samples
    """


    return_tensor = False
    if type(arr) == numpy.ndarray:
        pass
    else:
        arr = arr.cpu().numpy()
        return_tensor = True
        arr = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))

    _, idxs = np.unique(arr, return_index=True)

    if torch.cuda.is_available() and return_tensor:
        return torch.LongTensor(np.sort(idxs)).cuda()
    elif return_tensor:
        return torch.LongTensor(np.sort(idxs))
    else:
        return np.sort(idxs)


def padding_and_valid_mask(sequences, action_space, max_seq_length):
    """
    Pads a list of sequences to the maximum sequence length and creates a valid mask.
    Args:
        sequences: (torch.Tensor)
        action_space:
        max_seq_length:

    Returns:
        padded_sequences: (batch_size, max_seq_length) *Tensor of padded action sequences*
        valid_mask: (batch_size, max_seq_length) *Mask indicating valid steps in the sequences (including one end token)*

    """
    device = sequences[0].device
    batch_size = len(sequences)
    end_token_index = action_space.reversed_action_space['End']
    padded_sequences = torch.full(
        (batch_size, max_seq_length),
        fill_value=end_token_index,
        device=device,
        dtype=torch.long
    )
    for i, seq in enumerate(sequences):
        copy_len = min(len(seq), max_seq_length)
        padded_sequences[i, :copy_len] = seq[:copy_len]

    is_end_token = (padded_sequences == end_token_index)
    first_end_indices = torch.argmax(is_end_token.int(), dim=1)
    lengths = torch.where(
        is_end_token.any(dim=1) & (first_end_indices > 0),
        first_end_indices + 1,
        max_seq_length
    )
    arange_tensor = torch.arange(max_seq_length, device=device)
    valid_mask = arange_tensor < lengths.unsqueeze(1)

    return padded_sequences, valid_mask


if __name__ == "__main__":

    grammar_path = "./data/test_grammar_1.txt"
    grammar = GroupGrammar.from_file(grammar_path)
    action_space = Action_Space(grammar)
    action_sequence = [14,0, 12]
    out = "[:0toluene][Ring1][pop][=Branch][:0trifluoromethane][pop][=Branch][:1sulfonamide][C][:0pyrazole][Ring1]"
    sequence = action_space.encoding_to_action_sequence(out)
    out_new = action_space.action_sequence_to_encoding(sequence)
    print("--")
    print(out)
    print(sequence)
    print(out_new)




