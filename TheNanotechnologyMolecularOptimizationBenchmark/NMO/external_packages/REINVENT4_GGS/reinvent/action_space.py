from group_selfies import constants as gs_const
from group_selfies import GroupGrammar
import numpy as np
import  re
from abc import ABC, abstractmethod

class Action_Space(ABC):
    """
    Abstract class to handle the action space of the RNN.
    """
    def __init__(self):
        pass

    @property
    @abstractmethod
    def action_space(self):
        """
        Returns the action space as a dictionary mapping action indices to action names.
        """
        pass

    @property
    @abstractmethod
    def reversed_action_space(self):
        """
        Returns the reversed action space as a dictionary mapping action names to action indices.
        """
        pass

    @property
    @abstractmethod
    def N_actions(self):
        """
        Returns the number of actions in the action space.
        """
        pass

    @abstractmethod
    def action_sequence_to_encoding(self, action_sequence):
        """
        Convert an action sequence to the specific encoding.
        Args:
            action_sequence : array of action indices
        Returns:
            encoding : str, encoded action sequence
        """
        pass

    @abstractmethod
    def encoding_to_action_sequence(self, encoding):
        """
        Convert an encoded action sequence to the action sequence.
        Args:
            encoding : str, encoded action sequence
        Returns:
            action_sequence : array of action indices
        """
        pass

    def has_end_token(self, action_sequence):
        """
        Check if the action sequence has an end token. And make sure no actions are taken after the end token.
        Args:
            action_sequence : list of action indices
        Returns:
            has_end : bool, whether the action sequence has an end token
        """
        end_token_index = self.reversed_action_space["End"]
        end_token_found = False
        for action in action_sequence:
            if action == end_token_index:
                end_token_found = True
            if end_token_found and action != end_token_index:
                return False

        return end_token_found





class Action_Space_GroupSelfies(Action_Space):
    def __init__(self, selfies_grammar):
        super().__init__()

        self.selfies_grammar = selfies_grammar
        self.groups = list(self.selfies_grammar.vocab)
        n_attach_max = max([len(g.attachment_points) for g in selfies_grammar.vocab.values()])
        #this should be set to the maximum number of attachment points found in a group
        self.coupling_in = np.arange(0,n_attach_max)
        self.coupling_in = [str(item) for item in self.coupling_in]
        self.shift_out = [item.replace("[","").replace("]","") for item in(gs_const.INDEX_ALPHABET[0:n_attach_max])]
        #make sure no group is named like an item in gs_const.INDEX_ALPHABET
        if any(item in selfies_grammar.vocab for item in self.shift_out):
            raise ValueError("Group name cannot be same as index alphabet")
        self.special_tokens = ["Start", "pop", "End"]

        #add groups, coupling_in and shift_out to one list
        action_space_ = self.groups + self.coupling_in + list(self.shift_out) + self.special_tokens
        #reversed_action_space gives index from action
        self.reversed_action_space_ = dict(zip(action_space_, range(len(action_space_))))
        #action_space gives action from index
        self.action_space_ = {v: k for k, v in self.reversed_action_space_.items()}

        #print("Action space", action_space_)
        #print("Reversed action space", self.reversed_action_space_)

        self.N_actions_ = len(self.reversed_action_space_)

    @classmethod
    def from_grammar_path(cls, grammar_path):
        """
        Create an action space from a grammar path
        :param grammar_path:
        :return:
        """
        selfies_grammar = GroupGrammar.from_file(grammar_path)
        return cls(selfies_grammar)

    @property
    def action_space(self):
        """
        Returns the action space as a dictionary mapping action indices to action names.
        """
        return self.action_space_

    @property
    def reversed_action_space(self):
        """
        Returns the reversed action space as a dictionary mapping action names to action indices.
        """
        return self.reversed_action_space_

    @property
    def N_actions(self):
        """
        Returns the number of actions in the action space.
        """
        return self.N_actions_


    def get_action_type(self, one_hot_index):
        """
        Get the action type from the one hot index
        :param one_hot_index:
        :return:
        """
        if one_hot_index < len(self.groups):
            return "group"
        elif one_hot_index >= len(self.groups) and one_hot_index < len(self.groups) + len(self.coupling_in):
            return "coupling_in"
        elif one_hot_index >= len(self.groups) + len(self.coupling_in) and one_hot_index < len(self.groups) + len(self.coupling_in) + len(self.shift_out):
            return "shift_out"
        elif one_hot_index >= len(self.groups) + len(self.coupling_in) + len(self.shift_out):
            return "special_tokens"


    def action_sequence_to_encoding(self, action_sequence):
        """
        Convert an action sequence to a group selfies
        :param action_sequence: array of action indices
        :return:
        """
        group_selfies = ""
        for action in action_sequence:
            action_type = self.get_action_type(action)
            if action_type == "special_tokens" and self.action_space[action] == "End":
                break
            if action_type == "group":
                group_selfies += f"{self.action_space[action]}]"
            elif action_type == "coupling_in":
                group_selfies += f"[:{self.action_space[action]}"
            elif action_type == "shift_out":
                group_selfies += f"[{self.action_space[action]}]"
            elif action_type == "special_tokens":
                group_selfies += f"[{self.action_space[action]}]"
        return group_selfies

    def encoding_to_action_sequence(self, group_selfies):
        """
        Convert a group selfies to an action sequence.
        Args:
            group_selfies : str, group selfies encoding
        Returns:
            action_sequence : list of action indices
        """

        action_sequence = []
        group_selfies_split = [s for s in re.split(r'\[|\]', group_selfies) if s]
        for part in group_selfies_split:
            if ":" in part:
                match = re.search(r':\d+', part)
                if match:
                    index = match.end()
                    group = part[index:]
                    coupling_in = part[part.index(":")+1:index]
                    action_sequence.append(self.reversed_action_space[str(coupling_in)])
                    try:
                        action_sequence.append(self.reversed_action_space[group])
                    except KeyError as e:
                        print(f"Group '{group}' not found in action space.")
                        action_sequence = []
                        return action_sequence

                else:
                    raise ValueError("Group selfies is not valid")
            else:
                action_sequence.append(self.reversed_action_space[part])
        action_sequence.append(self.reversed_action_space["End"])

        return action_sequence




class Action_Space_Smiles(Action_Space):
    """A class for handling encoding/decoding from SMILES to an array of indices
    This class is adapted from https://github.com/hyeonahkimm/genetic_gfn/blob/main/sars_cov2/genetic_gfn/data_structs.py
    """
    def __init__(self, init_from_file=None, max_length=140):
        self.special_tokens = ['Start', 'End']
        self.additional_chars = set()
        self.chars = self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.max_length = max_length
        if init_from_file: self.init_from_file(init_from_file)

    @property
    def action_space(self):
        return self.reversed_vocab

    @property
    def reversed_action_space(self):
        return self.vocab

    @property
    def N_actions(self):
        """Returns the number of actions in the action space."""
        return self.vocab_size

    def encoding_to_action_sequence(self, encoding):
        """
        Takes a SMILES string and returns an array of indices corresponding to the characters in the vocabulary.
        Args:
            encoding : str, SMILES string to be encoded
        Returns:
            smiles_matrix : np.array, array of indices corresponding to the characters in the vocabulary
        """

        char_list = self.tokenize(encoding)
        end_token_index = self.reversed_action_space["End"]
        smiles_matrix = np.full(len(char_list),end_token_index, dtype=np.float32)
        for i, char in enumerate(char_list):
            try:
                smiles_matrix[i] = self.vocab[char]
            except KeyError as e:
                raise KeyError(f"Character '{char}' in {char_list} not found in vocabulary. Please add it to the vocabulary.") from e
        return smiles_matrix



    def action_sequence_to_encoding(self, matrix):
        """Takes an array of indices and returns the corresponding SMILES"""
        chars = []
        for i in matrix:
            if i == self.vocab['End']: break
            chars.append(self.reversed_vocab[i])
        smiles = "".join(chars)
        smiles = smiles.replace("L", "Cl").replace("R", "Br")
        return smiles

    def tokenize(self, smiles):
        """Takes a SMILES and return a list of characters/tokens"""
        regex = '(\[[^\[\]]{1,6}\])'
        smiles = self.replace_halogen(smiles)
        char_list = re.split(regex, smiles)
        tokenized = []
        for char in char_list:
            if char.startswith('['):
                tokenized.append(char)
            else:
                chars = [unit for unit in char]
                [tokenized.append(unit) for unit in chars]
        tokenized.append('End')
        return tokenized

    def add_characters(self, chars):
        """Adds characters to the vocabulary"""
        for char in chars:
            self.additional_chars.add(char)
        char_list = list(self.additional_chars)
        char_list.sort()
        self.chars = char_list + self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}

    def init_from_file(self, file):
        """Takes a file containing \n separated characters to initialize the vocabulary"""
        with open(file, 'r') as f:
            chars = f.read().split()
        self.add_characters(chars)

    def __len__(self):
        return len(self.chars)

    def __str__(self):
        return "Vocabulary containing {} tokens: {}".format(len(self), self.chars)

    def replace_halogen(self, string):
        """Regex to replace Br and Cl with single letters"""
        br = re.compile('Br')
        cl = re.compile('Cl')
        string = br.sub('R', string)
        string = cl.sub('L', string)

        return string

    def has_end_token(self, action_sequence):
        """
        Check if the action sequence has an end token. -> For smiles this is bypassed
        Args:
            action_sequence : list of action indices
        Returns:
            has_end : bool, whether the action sequence has an end token
        """
        return True


if __name__ == "__main__":
    data_path = "./data/smiles_voc.dat"
    action_space_smiles = Action_Space_Smiles(data_path)
    test = action_space_smiles.reversed_action_space['Start']
    print(test)