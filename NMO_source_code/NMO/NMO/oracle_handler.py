import configparser
import copy
import hashlib
import multiprocessing
import time
from abc import ABC, abstractmethod
import numpy as np
import torch
from collections import defaultdict
from rdkit import Chem
import os
from group_selfies import GroupGrammar
import re
from rdkit.Contrib.SA_Score import sascorer

from GGS import disable_rdkit_logging, GGS, validate_molecule

from .transport_workflow_handler import transport_workflow_handler, _write_candidate_metadata
from .terahertz_work_flow_handler import terahertz_workflow_handler



__dtype__ = torch.float64

class Oracle_Handler(ABC):
    def __init__(self, config_file, rng = None):

        #check if self.config was already defined in child class
        if not hasattr(self, 'config'):
            self.config_file = config_file
            self.config = configparser.ConfigParser()
            self.config.read(config_file, encoding='utf-8')

        self.grammar_path = self.config.get("General", "grammar_path", fallback=None)
        self.n_process = self.config.getint("Training", "n_oracle_processes", fallback=1)
        #string to identify the fitness function
        self.fitness_func = self.config.get("Oracle", "fitness_func")
        #calculated properties -> can be "SA", "PF", "tdc.xxx" or a combination of these (comma separated)
        self.calculated_props = self.config.get("Oracle", "calculated_props")

        self.max_oracle_calls = self.config.getint("Oracle", "max_oracle_calls", fallback=-1)
        self.oracle_calls = 0
        self.rng = rng



    def get_rewards(self, encoding_batch, meta_data = {}):
        """
        Calculate the rewards (not fitness!) for a batch of encoded molecules using multiprocessing.
        Args:
            encoding_batch : list of encoding
            meta_data : dict, additional information about the batch (e.g. step) for logging purposes
        """

        #this can be use once dxtb is fully functional. right now we parallelize over workflow_handler if needed
        #is_expensive = "ph_transp" in self.calculated_props or "el_transp" in self.calculated_props
        is_expensive = False

        if not is_expensive:
            list_of_dicts = [self.get_rewards_subproc(0, encoding_batch, meta_data)]
            processing_indices = [list(range(len(encoding_batch)))]
        else:
            # Check if self.groups_n_atoms exists
            if hasattr(self, 'groups_n_atoms'):
                def get_n_atoms(group_selfies):
                    group_selfies_split = [s for s in re.split(r'\[|\]', group_selfies) if s]
                    groups = []
                    for part in group_selfies_split:
                        if ":" in part:
                            match = re.search(r':\d+', part)
                            if match:
                                index = match.end()
                                group = part[index:]
                                groups.append(group)
                    n_atoms = []
                    for item in groups:
                        if item in self.grammar.vocab and hasattr(self.grammar.vocab[item], 'mol'):
                            n_atoms.append(self.grammar.vocab[item].mol.GetNumAtoms())
                        else:
                            n_atoms.append(0)  # Or use 0, or skip this group

                    return np.sum(n_atoms) if n_atoms else 0

                n_atoms = np.array([get_n_atoms(encoding) for encoding in encoding_batch])

                # Load balance by atom count while preserving order
                indices = np.argsort(n_atoms)[::-1]  # Sort by size, largest first
                processing_parts = [[] for _ in range(self.n_process)]
                processing_indices = [[] for _ in range(self.n_process)]
                load_per_process = np.zeros(self.n_process)

                # Greedy bin packing: assign each molecule to least loaded process
                for idx in indices:
                    min_load_proc = np.argmin(load_per_process)
                    processing_parts[min_load_proc].append(encoding_batch[idx])
                    processing_indices[min_load_proc].append(idx)
                    load_per_process[min_load_proc] += n_atoms[idx]
            else:
                # Fallback to simple splitting if groups_n_atoms not available
                processing_parts = np.array_split(encoding_batch, self.n_process)
                processing_indices = [list(range(len(part))) for part in processing_parts]
                processing_parts = [part.tolist() for part in processing_parts]
            multiprocessing.set_start_method('spawn', force=True)
            starmap_args = [(i, part if isinstance(part, list) else part.tolist(), meta_data)
                            for i, part in enumerate(processing_parts)]
            with multiprocessing.Pool(processes=self.n_process) as pool:
                list_of_dicts = pool.starmap(self.get_rewards_subproc, starmap_args)

        # Merge results while preserving original order
        merged_rewards = defaultdict(lambda: [0.0] * len(encoding_batch))
        for proc_idx, result_dict in enumerate(list_of_dicts):
            original_indices = processing_indices[proc_idx]
            for key, value_list in result_dict.items():
                for i, orig_idx in enumerate(original_indices):
                    merged_rewards[key][orig_idx] = value_list[i]

        # Convert to numpy arrays
        for key in merged_rewards.keys():
            merged_rewards[key] = np.array(merged_rewards[key])
        merged_rewards.default_factory = lambda: np.zeros(len(encoding_batch))
        return merged_rewards

    def get_fitness(self, encoding_batch, meta_data = {}):
        """
        Calculate the fitness for a batch of encodings.
        Args:
            encoding_batch : list of encodings
            meta_data : dict, additional information about the batch (e.g. step) for logging purposes
        Returns:
            fitness : np.array of fitness values
            calculated_rewards : dict (keys: "SA", "PF") with np.array of scores
            oracle_calls_exceeded : bool, True if the maximum number of oracle calls is exceeded
        """
        meta_data["oracle_call_start"] = self.oracle_calls
        calculated_rewards = self.get_rewards(encoding_batch, meta_data)
        #add the number of oracle calls
        oracle_calls = np.arange(self.oracle_calls, self.oracle_calls + len(encoding_batch), dtype=int)
        calculated_rewards["oracle_calls"] = oracle_calls

        eval_env = {"np": np}
        eval_env.update(calculated_rewards)
        try:
            fitness = eval(self.fitness_func, {}, eval_env)
        except KeyError as e:
            print("Error calculating fitness function. Check if all required properties are calculated. Missing key:", e)
            fitness = np.zeros(len(encoding_batch))
        except NameError as e:
            print("Error calculating fitness function. Check if all required properties are calculated. Missing key:", e)
            fitness = np.zeros(len(encoding_batch))
        fitness = np.array(fitness)
        fitness = np.nan_to_num(fitness, nan=0.0, posinf=0.0, neginf=0.0)

        if "ph_transp" in self.calculated_props or "el_transp" in self.calculated_props or "P_upconversion" in self.calculated_props:
            log_dir = self.config.get("Training", "log_dir")
            worker_id = os.getpid()
            h5_filename = f"worker_{worker_id}_data.h5"
            h5_filepath = os.path.join(log_dir, h5_filename)
            failure_reasons = calculated_rewards.get("failure_reasons", [""] * len(encoding_batch))
            _write_candidate_metadata(h5_filepath, calculated_rewards["hash_values"], encoding_batch,
                                      failure_reasons, calculated_rewards, meta_data, fitness=fitness)

        self.oracle_calls += len(fitness)
        if self.max_oracle_calls != -1:
            oracle_calls_exceeded = self.oracle_calls > self.max_oracle_calls
        else:
            oracle_calls_exceeded = False



        return fitness, calculated_rewards, oracle_calls_exceeded


    @abstractmethod
    def get_rewards_subproc(self, worker_id, encoding_batch, meta_data = {}):
        """
        Calculate the rewards for a batch of encodings.
        Args:
            worker_id : int, id of the worker process
            encoding_batch : list of encodings
            meta_data : dict, additional information about the batch (e.g. step) for logging purposes
        Returns:
            scores : np.array of scores
        """
        pass


class Oracle_Handler_GGS(Oracle_Handler):
    def __init__(self, config_file, rng = None):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.config.read(config_file, encoding='utf-8')
        self.grammar_path = self.config.get("General", "grammar_path")
        self.grammar = GroupGrammar.from_file(self.grammar_path)
        self.groups = list(self.grammar.vocab)
        self.groups_n_atoms = np.array([self.grammar.vocab[item].mol.GetNumAtoms() for item in self.groups])
        super().__init__(config_file, rng)



    def get_rewards_subproc(self, worker_id, group_selfies_batch, meta_data = {}):
        """
        Calculate the rewards specified in mode for a batch of group selfies.
        Args:
            worker_id: Id of worker
            group_selfies_batch : list of group selfies
            calculated_props : str of calculated properties -> multiple properties can be combined when comma separated
            meta_data: dict, additional information about the batch (e.g. step) for logging purposes
        Returns:
            calculated_rewards : dict (keys: "SA", "PF") with np.array of scores
        """
        calculated_rewards = {}

        log_dir = self.config.get("Training", "log_dir")
        save_xyz = False
        dir = None
        if "debug" in list(meta_data.keys()) and meta_data["debug"] == True:
            if "generation" in list(meta_data.keys()):
                dir = f"{log_dir}/debug/step_{meta_data['step']}_generation_{meta_data['generation']}_worker_id_{worker_id}"
            else:
                dir = f"{log_dir}/debug/step_{meta_data['step']}_worker_id_{worker_id}"
            if not os.path.exists(dir):
                pass
                #os.makedirs(dir, exist_ok=True)
            #save_xyz = True

        #prepare smiles codes and mol objects, create geometries where needed
        smiles = []
        smiles_3d = []
        smiles_3d_SA = []
        mols = []
        mols_3d = []
        mols_3d_SA = []
        ggs_list = []
        anchors = []
        valid_batches_mask = np.ones(len(group_selfies_batch), dtype=bool)
        batch_indices = np.arange(len(group_selfies_batch))
        failure_reasons = [""] * len(group_selfies_batch)

        #check if 3d structures have to be created
        geometry_created = False
        if "ph_transp" in self.calculated_props or "el_transp" in self.calculated_props or "P_upconversion" in self.calculated_props:
            geometry_created = True

        for i in range(len(group_selfies_batch)):
            try:
                ggs = GGS(group_selfies_batch[i], self.grammar, self.rng)

                if "ph_transp" in self.calculated_props or "el_transp" in self.calculated_props:
                    if "P_upconversion" in self.calculated_props:
                        raise ValueError("P_upconversion and ph_transp cannot be calculated together due to different anchor modes.")
                    mol_3d, _ = ggs.create_3d_structure(save_xyz=save_xyz, save_path=f"{dir}/debug_{i}.xyz", anchor_mode="AuS")

                if "P_upconversion" in self.calculated_props:
                    current_anchors = None
                    if "el_transp" in self.calculated_props or "ph_transp" in self.calculated_props:
                        raise ValueError("P_upconversion and ph_transp or el_transp cannot be calculated together due to different anchor modes.")
                    mol_3d, current_anchors = ggs.create_3d_structure(save_xyz=save_xyz, save_path=f"{dir}/debug_{i}.xyz", anchor_mode="AuS_just_left")

                if geometry_created:
                    #create mol object without gold for proper SA analysis
                    rw_mol_SA = Chem.RWMol(mol_3d)

                    for atom in rw_mol_SA.GetAtoms():
                        if atom.GetSymbol() == 'Au':
                            atom.SetAtomicNum(1)
                    mol_3d_SA = rw_mol_SA.GetMol()
                    try:
                        Chem.SanitizeMol(mol_3d_SA)
                    except Exception:
                        # Fallback if sanitization fails
                        Chem.GetSSSR(mol_3d_SA)

                    smile_3d = Chem.MolToSmiles(mol_3d)
                    smile_3d_SA = Chem.MolToSmiles(mol_3d_SA)

                    # filters are applied but they should be already filtered beforhand in train
                    keep, reason = validate_molecule(smile_3d_SA)
                    if not keep:
                        raise ValueError(f"Invalid molecule due to: {reason}")

                    #check if molecule has charge
                    formal_charge = Chem.GetFormalCharge(Chem.MolFromSmiles(smile_3d))
                    if formal_charge != 0:
                        raise ValueError(
                            "Charged molecules are not allowed when calculating ph_transp, el_transp or P_upconversion")


                mol = ggs.mol
                id = Chem.AllChem.EmbedMolecule(mol, randomSeed=0)
                if id == -1:
                    raise ValueError("Could not embed molecule")
                Chem.SanitizeMol(mol)
                smiles_ = Chem.MolToSmiles(mol)
                smiles.append(smiles_)
                mols.append(mol)
                if geometry_created:
                    mols_3d.append(mol_3d)
                    mols_3d_SA.append(mol_3d_SA)
                    smiles_3d.append(smile_3d)
                    smiles_3d_SA.append(smile_3d_SA)
                    if "P_upconversion" in self.calculated_props:
                        anchors.append(current_anchors)


                ggs_list.append(ggs)

            except Exception as e:
                valid_batches_mask[i] = 0
                failure_reasons[i] = str(e)
                smiles_ = ""
                smiles.append(smiles_)
                mol = Chem.MolFromSmiles(smiles_)
                mols.append(mol)
                ggs_list.append(None)
                if geometry_created:
                    mols_3d.append(mol)
                    mols_3d_SA.append(mol)
                    smiles_3d.append(smiles_)
                    smiles_3d_SA.append(smiles_)
                    anchors.append([-1, -1])

        assert len(mols) == len(smiles) == len(ggs_list), f"Length of mols, smiles and ggs_list must be equal, but found {len(mols)} and {len(smiles)} and {len(ggs_list)}"

        smiles = np.array(smiles)
        mols = np.array(mols)
        ggs_list = np.array(ggs_list)
        if len(anchors) > 0:
            anchors = np.array(anchors)

        if geometry_created:
            assert len(ggs_list) == len(mols_3d)
            mols_3d = np.array(mols_3d)
            mols_3d_SA = np.array(mols_3d_SA)
            smiles_3d = np.array(smiles_3d)
            smiles_3d_SA = np.array(smiles_3d_SA)
            calculated_rewards["smiles"] = smiles_3d
        else:
            calculated_rewards["smiles"] = smiles

        batch_indices = batch_indices[valid_batches_mask]

        if "SA" in self.calculated_props and "tdc_" not in self.calculated_props:
            sa_scores = np.ones(len(group_selfies_batch)) * 10
            for idx in batch_indices:
                if geometry_created:
                    # clean history of mol
                    smiles_tmp = Chem.MolToSmiles(mols_3d_SA[idx])
                    mol_tmp = Chem.MolFromSmiles(smiles_tmp)
                    sa_scores[idx] = sascorer.calculateScore(mol_tmp)
                else:
                    sa_scores[idx] = sascorer.calculateScore(mols[idx])
            calculated_rewards["SA"] = sa_scores


        if "tdc_" in self.calculated_props:
            split = self.calculated_props.strip().split(",")
            tdc_props = [prop.replace("tdc_", "") for prop in split if prop.startswith("tdc_")]

            for prop in tdc_props:
                try:
                    from tdc import Oracle
                    oracle = Oracle(name=prop)
                    scores = np.zeros(len(group_selfies_batch))
                    if prop == "SA" and mols_3d_SA.shape[0] > 0:
                        scores[batch_indices] = np.array(oracle(smiles_3d_SA[batch_indices].tolist()))
                    else:
                        scores[batch_indices] = np.array(oracle(smiles[batch_indices].tolist()))

                    calculated_rewards["tdc_" + prop] = np.array(scores)

                except Exception as e:
                    print(f"Error calculating TDC property {prop}: {e}")
                    calculated_rewards["tdc_" + prop] = np.zeros(len(group_selfies_batch))

        if "N_rot" in self.calculated_props:
            n_rot_bonds = np.ones(len(group_selfies_batch), dtype=int)
            for idx in batch_indices:
                mol = mols[idx]
                rot_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
                n_rot_bonds[idx] = rot_bonds
            calculated_rewards["N_rot"] = n_rot_bonds

        if geometry_created:
            key_to_be_filled = ["hash_values", "smiles", "hl_gaps"]
            if "ph_transp" in self.calculated_props:
                key_to_be_filled.append("k_ph")
            if "el_transp" in self.calculated_props:
                key_to_be_filled.append("G")
                key_to_be_filled.append("S")
                key_to_be_filled.append("k_el")
            if "P_upconversion" in self.calculated_props:
                key_to_be_filled.append("P_upconversion")

            #filter for valid batches
            if len(batch_indices) == 0:
                timestamp_ns = time.time_ns()
                for key in key_to_be_filled:
                    if key == "hash_values":
                        calculated_rewards[key] = np.array([
                            hashlib.md5(f"{enc}|{timestamp_ns}|{i}".encode("utf-8")).hexdigest()
                            for i, enc in enumerate(group_selfies_batch)
                        ])
                    elif key not in calculated_rewards.keys():
                        calculated_rewards[key] = [0] * len(group_selfies_batch)
                    else:
                        calculated_rewards[key] = np.array([""] * len(group_selfies_batch))
                calculated_rewards["failure_reasons"] = np.array(failure_reasons, dtype=object)
                return calculated_rewards

            config_dict = {}
            config_dict["calculated_props"] = self.calculated_props
            config_dict["log_dir"] = log_dir
            config_dict["key_to_be_filled"] = key_to_be_filled
            #config_dict["gradient_tolerance"] = self.config.getfloat("Oracle", "grad_tolerance")
            #config_dict["fmax_tolerance"] = self.config.getfloat("Oracle", "fmax_tolerance")
            #config_dict["optimization_steps"] = self.config.getint("Oracle", "optimization_steps")
            config_dict["memory_fraction_limit"] = 0.95 / self.n_process
            tmp = self.config.getint("Oracle", "n_cpus_total", fallback=-1)
            if tmp != -1:
                config_dict["n_cpus"] = tmp


            if "el_transp" in self.calculated_props:
                config_dict["E_min"] = self.config.getfloat("Oracle", "E_min", fallback = -18)
                config_dict["E_max"] = self.config.getfloat("Oracle", "E_max", fallback = -6)
                config_dict["N_E_points"] = self.config.getint("Oracle", "N_E_points", fallback = 1000)
                config_dict["electrode_path"] = self.config.get("Oracle", "electrode_path")

            if "P_upconversion" in self.calculated_props:
                config_dict["w_area"] = self.config.getfloat("Oracle", "w_area", fallback=1)
                config_dict["w_len"] = self.config.getfloat("Oracle", "w_len", fallback=1)

            if "el_transp" in self.calculated_props or "ph_transp" in self.calculated_props:
                calculated_rewards, batch_indices_wf_handler, wf_failure_reasons = transport_workflow_handler(mols_3d, group_selfies_batch, config_dict, meta_data, calculated_rewards, batch_indices, worker_id = worker_id)
            elif "P_upconversion" in self.calculated_props:
                calculated_rewards, batch_indices_wf_handler, wf_failure_reasons = terahertz_workflow_handler(mols_3d, group_selfies_batch, config_dict, meta_data, calculated_rewards, batch_indices, anchor_atoms=anchors, worker_id = worker_id)
            merged_dict = {}
            full_batch_keys = {"hash_values", "failure_reasons"}

            for key, val in calculated_rewards.items():
                if key in full_batch_keys:
                    merged_dict[key] = np.array(val)
                    continue

                val = np.array(val)
                if np.issubdtype(val.dtype, np.number):
                    values = np.zeros(len(group_selfies_batch), dtype=val.dtype)
                else:
                    values = np.full(len(group_selfies_batch), "", dtype=object)

                values[batch_indices_wf_handler] = val[batch_indices_wf_handler]
                merged_dict[key] = values

            calculated_rewards = merged_dict

            for i, reason in enumerate(wf_failure_reasons):
                if reason:
                    failure_reasons[i] = reason
            calculated_rewards["failure_reasons"] = np.array(failure_reasons, dtype=object)

        return calculated_rewards


class Oracle_Handler_Smiles(Oracle_Handler):
    def __init__(self, config_file, rng = None):
        super().__init__(config_file, rng)

    def get_rewards(self, encoding_batch, meta_data = {}, anchor_atoms = None):
        _meta = dict(meta_data)
        if anchor_atoms is not None:
            _meta["anchor_atoms"] = anchor_atoms
        return super().get_rewards(encoding_batch, _meta)

    def get_fitness(self, encoding_batch, meta_data = {}, anchor_atoms = None):
        _meta = dict(meta_data)
        if anchor_atoms is not None:
            _meta["anchor_atoms"] = anchor_atoms
        return super().get_fitness(encoding_batch, _meta)

    def get_rewards_subproc(self, worker_id, smiles_batch, meta_data = {}):
        """
        Calculate the scores for a batch of group selfies.
        Args:
            worker_id: Id of worker
            smiles_batch : list of group selfies
            meta_data : dict, additional information about the batch (e.g. step) for logging purposes
        Returns:
            calculated_rewards : dict (keys: "SA") with np.array of scores
        """
        calculated_rewards = {}

        # prepare smiles codes and mol objects, create geometries where needed
        smiles_3d = []
        smiles_3d_SA = []
        mols = []
        mols_3d = []
        mols_3d_SA = []
        anchors = []
        valid_batches_mask = np.ones(len(smiles_batch), dtype=bool)
        batch_indices = np.arange(len(smiles_batch))
        failure_reasons = [""] * len(smiles_batch)
        geometry_created = False
        if "ph_transp" in self.calculated_props or "el_transp" in self.calculated_props or "P_upconversion" in self.calculated_props:
            geometry_created = True

        for i in range(len(smiles_batch)):
            try:
                mol = Chem.MolFromSmiles(smiles_batch[i])
                # Add S-Au Anchors to beginning and end
                if geometry_created:
                    #check if sulfur already in mol -> this is not valid because the junction formation cannot be
                    #controlled -> check mol object
                    has_sulfur = any(atom.GetAtomicNum() == 16 for atom in mol.GetAtoms())
                    if has_sulfur:
                        raise ValueError("Sulfur not allowed in input molecule when calculating ph_transp, el_transp or P_upconversion")

                    num_atoms = mol.GetNumAtoms()
                    _anchor_atoms = meta_data.get("anchor_atoms")
                    if _anchor_atoms is not None:
                        _pair = _anchor_atoms if not isinstance(_anchor_atoms[0], (list, tuple, type(None))) else _anchor_atoms[i]
                        if _pair is None:
                            first_atom_idx, last_atom_idx = 0, num_atoms - 1
                        else:
                            first_atom_idx, last_atom_idx = int(_pair[0]), int(_pair[1])
                            if not (0 <= first_atom_idx < num_atoms) or not (0 <= last_atom_idx < num_atoms):
                                raise ValueError(f"Anchor indices [{first_atom_idx}, {last_atom_idx}] out of range for molecule with {num_atoms} atoms")
                    else:
                        first_atom_idx, last_atom_idx = 0, num_atoms - 1
                    em = Chem.EditableMol(mol)
                    s1_idx = em.AddAtom(Chem.Atom(16))  # Sulfur (Atomic Num 16)
                    au1_idx = em.AddAtom(Chem.Atom(79))  # Gold (Atomic Num 79)
                    em.AddBond(first_atom_idx, s1_idx, Chem.BondType.SINGLE)
                    em.AddBond(s1_idx, au1_idx, Chem.BondType.SINGLE)
                    first_atom_idx = au1_idx

                    if "P_upconversion" not in self.calculated_props:
                        s2_idx = em.AddAtom(Chem.Atom(16))  # Sulfur
                        au2_idx = em.AddAtom(Chem.Atom(79))  # Gold
                        # Bond original last atom to new sulfur
                        em.AddBond(last_atom_idx, s2_idx, Chem.BondType.SINGLE)

                        # Bond new sulfur to new gold
                        em.AddBond(s2_idx, au2_idx, Chem.BondType.SINGLE)

                        last_atom_idx = au2_idx

                    mol_3d = em.GetMol()
                    Chem.SanitizeMol(mol_3d)
                    mol_3d = Chem.AddHs(mol_3d)

                    # create mol object without gold for proper SA analysis
                    rw_mol_SA = Chem.RWMol(mol_3d)
                    for atom in rw_mol_SA.GetAtoms():
                        if atom.GetSymbol() == 'Au':
                            atom.SetAtomicNum(1)
                    mol_3d_SA = rw_mol_SA.GetMol()

                    try:
                        Chem.SanitizeMol(mol_3d_SA)
                    except Exception:
                        # Fallback if sanitization fails
                        Chem.GetSSSR(mol_3d_SA)

                    smile_3d = Chem.MolToSmiles(mol_3d)
                    smile_3d_SA = Chem.MolToSmiles(mol_3d_SA)

                    #filters are applied but they should be already filtered beforhand in train
                    keep, reason = validate_molecule(smile_3d_SA)
                    if not keep:
                        raise ValueError(f"Invalid molecule due to: {reason}")

                    id = Chem.AllChem.EmbedMolecule(mol_3d, randomSeed=0)
                    if id == -1:
                        raise ValueError("Could not embed molecule")

                    #check if molecule has charge
                    formal_charge = Chem.GetFormalCharge(Chem.MolFromSmiles(smile_3d))
                    if formal_charge != 0:
                        raise ValueError("Charged molecules are not allowed when calculating ph_transp, el_transp or P_upconversion")


                    mols_3d.append(mol_3d)
                    mols_3d_SA.append(mol_3d_SA)
                    smiles_3d.append(smile_3d)
                    smiles_3d_SA.append(smile_3d_SA)
                    anchors.append([first_atom_idx, last_atom_idx])
                mols.append(mol)
            except Exception as e:
                valid_batches_mask[i] = 0
                failure_reasons[i] = str(e)
                mol = Chem.MolFromSmiles("")
                mols.append(mol)

                if geometry_created:
                    mols_3d.append(mol)
                    mols_3d_SA.append(mol)
                    smiles_3d.append("")
                    smiles_3d_SA.append("")
                    anchors.append([-1, -1])



        assert len(mols) == len(smiles_batch), f"Length of mols, smiles and ggs_list must be equal, but found {len(mols)} and {len(smiles_batch)}"

        smiles = np.array(smiles_batch)
        mols = np.array(mols)

        if len(anchors) > 0:
            anchors = np.array(anchors)

        if geometry_created:
            assert len(smiles_batch) == len(mols_3d)
            mols_3d = np.array(mols_3d)
            mols_3d_SA = np.array(mols_3d_SA)
            smiles_3d = np.array(smiles_3d)
            smiles_3d_SA = np.array(smiles_3d_SA)
            calculated_rewards["smiles"] = smiles_3d
        else:
            calculated_rewards["smiles"] = smiles

        batch_indices = batch_indices[valid_batches_mask]

        if "SA" in self.calculated_props and "tdc_" not in self.calculated_props:
            sa_scores = np.ones(len(smiles_batch)) * 10
            for idx in batch_indices:
                if geometry_created > 0:
                    #clean history of mol
                    smiles_tmp = Chem.MolToSmiles(mols_3d_SA[idx])
                    mol_tmp = Chem.MolFromSmiles(smiles_tmp)
                    sa_scores[idx] = sascorer.calculateScore(mol_tmp)
                else:
                    sa_scores[idx] = sascorer.calculateScore(mols[idx])
            calculated_rewards["SA"] = sa_scores


        if "tdc_" in self.calculated_props:
            split = self.calculated_props.strip().split(",")
            tdc_props = [prop.replace("tdc_", "") for prop in split if prop.startswith("tdc_")]
            for prop in tdc_props:
                try:
                    from tdc import Oracle
                    oracle = Oracle(name=prop)
                    if geometry_created:
                        scores = oracle([Chem.MolToSmiles(m) for m in mols_3d_SA])
                    else:
                        scores = oracle(list(smiles_batch))

                    calculated_rewards["tdc_" + prop] = np.array(scores)
                except Exception as e:
                    print(f"Error calculating TDC property {prop}: {e}")
                    calculated_rewards["tdc_" + prop] = np.zeros(len(smiles_batch))

        if "N_rot" in self.calculated_props:
            n_rot_bonds = np.ones(len(smiles_batch), dtype=int)
            for idx in batch_indices:
                mol = mols[idx]
                rot_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
                n_rot_bonds[idx] = rot_bonds
            calculated_rewards["N_rot"] = n_rot_bonds


        if geometry_created:
            key_to_be_filled = ["hash_values", "smiles", "hl_gaps"]
            if "ph_transp" in self.calculated_props:
                key_to_be_filled.append("k_ph")
            if "el_transp" in self.calculated_props:
                key_to_be_filled.append("G")
                key_to_be_filled.append("S")
                key_to_be_filled.append("k_el")
            if "P_upconversion" in self.calculated_props:
                if "ph_transp" in self.calculated_props or "el_transp" in self.calculated_props:
                    raise ValueError("P_upconversion cannot be calculated together with ph_transp or el_transp due to different anchor modes.")
                key_to_be_filled.append("P_upconversion")

            log_dir = self.config.get("Training", "log_dir")

            save_xyz = False
            dir = None
            if "debug" in list(meta_data.keys()) and meta_data["debug"] == True:
                if "generation" in list(meta_data.keys()):
                    dir = f"{log_dir}/debug/step_{meta_data['step']}_generation_{meta_data['generation']}_worker_id_{worker_id}"
                else:
                    dir = f"{log_dir}/debug/step_{meta_data['step']}_worker_id_{worker_id}"
                if not os.path.exists(dir):
                    pass
                    #os.makedirs(dir, exist_ok= True)
                save_xyz = True

            # filter for valid batches
            if len(batch_indices) == 0:
                timestamp_ns = time.time_ns()
                for key in key_to_be_filled:
                    if key == "hash_values":
                        calculated_rewards[key] = np.array([
                            hashlib.md5(f"{enc}|{timestamp_ns}|{i}".encode("utf-8")).hexdigest()
                            for i, enc in enumerate(smiles_batch)
                        ])
                    elif key not in calculated_rewards.keys():
                        calculated_rewards[key] = [0] * len(smiles_batch)
                    else:
                        calculated_rewards[key] = np.array([""] * len(smiles_batch))
                calculated_rewards["failure_reasons"] = np.array(failure_reasons, dtype=object)
                return calculated_rewards

            config_dict = {}
            config_dict["calculated_props"] = self.calculated_props
            config_dict["log_dir"] = log_dir
            config_dict["key_to_be_filled"] = key_to_be_filled
            tmp = self.config.getint("Oracle", "n_cpus_total", fallback=-1)
            if tmp != -1:
                config_dict["n_cpus"] = tmp

            if "el_transp" in self.calculated_props:
                config_dict["E_min"] = self.config.getfloat("Oracle", "E_min")
                config_dict["E_max"] = self.config.getfloat("Oracle", "E_max")
                config_dict["N_E_points"] = self.config.getint("Oracle", "N_E_points")
                config_dict["electrode_path"] = self.config.get("Oracle", "electrode_path")

            if "el_transp" in self.calculated_props or "ph_transp" in self.calculated_props:
                calculated_rewards, batch_indices_wf_handler, wf_failure_reasons = transport_workflow_handler(mols_3d, smiles_batch, config_dict, meta_data, calculated_rewards,
                                                  batch_indices, anchor_atom = 79, anchor_mode="")

            else:
                calculated_rewards, batch_indices_wf_handler, wf_failure_reasons = terahertz_workflow_handler(mols_3d, smiles_batch,
                                                                                          config_dict, meta_data,
                                                                                          calculated_rewards,
                                                                                          batch_indices,
                                                                                          anchor_atoms=anchors)

            merged_dict = {}
            full_batch_keys = {"hash_values", "failure_reasons"}

            for key, val in calculated_rewards.items():
                if key in full_batch_keys:
                    merged_dict[key] = np.array(val)
                    continue

                val = np.array(val)
                if np.issubdtype(val.dtype, np.number):
                    values = np.zeros(len(smiles_batch), dtype=val.dtype)
                else:
                    values = np.full(len(smiles_batch), "", dtype=object)

                values[batch_indices_wf_handler] = val[batch_indices_wf_handler]
                merged_dict[key] = values

            calculated_rewards = merged_dict

            for i, reason in enumerate(wf_failure_reasons):
                if reason:
                    failure_reasons[i] = reason
            calculated_rewards["failure_reasons"] = np.array(failure_reasons, dtype=object)

        return calculated_rewards





if __name__ == "__main__":
    gfs_1 = "[:1benzene][Branch][:0chlorine][pop][Ring2][:0carbon][Ring2][pop][Ring1][:0bromine][pop][#Branch][:1anthracene][#Branch][:1ethane][C][End]"
    gfs_2 = "[:1ethylene][C][:0ethylene][Ring1][:7anthracene][=Branch][:4naphthalene][Ring1][pop][C][:0ethylene][Ring1][:4naphthalene][#Branch][:1sulfonamide][C][:0naphthalene][Branch][End]"
    gfs_3 = "[:1benzene][Branch][:0chlorine][pop][Ring2][End]"
    gfs_4 = "[:1ethylene][C][End]"
    group_selfies_batch = [gfs_3, gfs_4, gfs_3, gfs_4]

    grammar_path = "./data/test_grammar_1.txt"
    oracle_handler = Oracle_Handler_GGS(grammar_path)

    scores = oracle_handler.get_rewards(group_selfies_batch, mode="PowerFactor")
    print(scores)


