#taken from https://github.com/hyeonahkimm/genetic_gfn/blob/main/sars_cov2/genetic_gfn/graph_ga_expert.py

from __future__ import print_function

import torch
import random
from typing import List
import copy

import numpy as np
from joblib import delayed, Parallel
from rdkit import Chem, rdBase
from rdkit.Chem.rdchem import Mol
from abc import ABC, abstractmethod
rdBase.DisableLog('rdApp.error')
from GGS import GGS
from genetic_operators_smiles import crossover, mutate
from functools import partial
from group_selfies import GroupGrammar

import gc

MINIMUM = 1e-10


class GeneticOperatorHandler(ABC):
    def __init__(self, config, seed = -1):
        """

        Args:
            config (ConfigParser): configuration parser
            seed:
        """
        self.config = config
        self.mutation_rate = config.getfloat("Genetic Search", "mutation_rate")
        self.population_size = config.getint("Genetic Search", "population_size")
        self.n_processes = config.getint("Training", "n_oracle_processes", fallback=1)
        self.seed = seed
        self.max_seq_length = config.getint("General", "max_seq_length")
        if self.seed != -1:
            self.generator = torch.Generator()
            self.generator.manual_seed(self.seed)
            self.rng = np.random.default_rng(seed=self.seed)
        else:
            self.rng = np.random.default_rng()

    @abstractmethod
    def query(self, query_size, mating_pool, rank_coefficient=0.01):
        """
        Query the genetic operator for new molecules.
        Args:
            query_size: int, number of molecules to generate
            mating_pool: tuple (list of encoding, list of rewards)
            pool: joblib Parallel pool for parallel processing
            rank_coefficient: float, coefficient for ranking
        """

    def select_pop(self, prev_elems, prev_scores, population_size, rank_coefficient=0.01, replacement=False):
        """
        Given a population of elements and their scores, sample a list of the same size
        with replacement using the scores as weights.
        Args:
            prev_elems: list of elements (e.g., RDKit Mol, group_selfies)
            prev_scores: list of un-normalised scores given by ScoringFunction
            population_size: number of elements to return
            rank_coefficient: float, coefficient for ranking
            replacement: bool, whether to sample with replacement
        Returns: list of sampled elements, list of their corresponding scores
        """
        scores_np = np.array(prev_scores)
        ranks = np.argsort(np.argsort(-1 * scores_np))
        weights = 1.0 / (rank_coefficient * len(scores_np) + ranks)

        if self.seed == -1:
            indices = list(torch.utils.data.WeightedRandomSampler(
                weights=weights, num_samples=population_size, replacement=replacement))
        else:
            indices = list(torch.utils.data.WeightedRandomSampler(
                weights=weights, num_samples=population_size, replacement=replacement, generator=self.generator))
        population_elems = [prev_elems[i] for i in indices]
        population_scores = [prev_scores[i] for i in indices]
        return population_elems, population_scores



class GeneticOperatorHandler_Smiles(GeneticOperatorHandler):
    """
    This class is adapted from https://github.com/hyeonahkimm/genetic_gfn/blob/main/sars_cov2/genetic_gfn/graph_ga_expert.py
    for benchmarking purposes.
    """
    def __init__(self, config, seed = -1):
        super().__init__(config, seed)


    def query(self, query_size, mating_pool, rank_coefficient=0.01):
        """
        Query the genetic operators for new molecules.
        Args:
            query_size: int, number of molecules to generate
            mating_pool: tuple (list of smiles, list of rewards)
            rank_coefficient: float, coefficient for ranking
        Returns
            smis: array SMILES strings of the generated molecules
            n_atoms: list of number of atoms in each generated molecule
            pop_valid_smis: list of valid SMILES strings from the mating pool
            pop_valid_scores: list of scores corresponding to the valid SMILES strings
        """


        population_mol = [Chem.MolFromSmiles(s) for s in mating_pool[0]]
        population_scores = mating_pool[1]

        cross_mating_pool, cross_mating_scores = self.make_mating_pool(population_mol, population_scores,
                                                                  self.population_size, rank_coefficient)

        n_jobs = 1
        pool = Parallel(n_jobs=n_jobs)
        offspring_mol = pool(delayed(self.reproduce)(cross_mating_pool) for _ in range(query_size))
        new_mating_pool = cross_mating_pool
        new_mating_scores = cross_mating_scores

        smis, n_atoms = [], []
        for m in offspring_mol:
            try:
                # smis.append(Chem.MolToSmiles(m))
                smi = Chem.MolToSmiles(m)
                if smi not in smis:  # unique
                    smis.append(smi)
                    n_atoms.append(m.GetNumAtoms())
            except:
                pass

        gc.collect()

        pop_valid_smis, pop_valid_scores = [], []
        for m, s in zip(new_mating_pool, new_mating_scores):
            try:
                # pop_valid_smis.append(Chem.MolToSmiles(m))
                pop_valid_smis.append(Chem.MolToSmiles(m))
                pop_valid_scores.append(s)
            except:
                pass
        smis = np.array(smis)

        crossover_stats_dummy = np.array([-1]*len(smis))
        mutation_stats_dummy = np.array([-1]*len(smis))

        return smis,crossover_stats_dummy, mutation_stats_dummy, pop_valid_smis, pop_valid_scores


    def make_mating_pool(self, population_mol: List[Mol], population_scores, population_size: int, rank_coefficient=0.01):
        """
        Given a population of RDKit Mol and their scores, sample a list of the same size
        with replacement using the population_scores as weights
        Args:
            population_mol: list of RDKit Mol
            population_scores: list of un-normalised scores given by ScoringFunction
            offspring_size: number of molecules to return
        Returns: a list of RDKit Mol (probably not unique)
        """
        # scores -> probs
        if rank_coefficient > 0:
            scores_np = np.array(population_scores)
            ranks = np.argsort(np.argsort(-1 * scores_np))
            weights = 1.0 / (rank_coefficient * len(scores_np) + ranks)
            if self.seed == -1:
                indices = list(torch.utils.data.WeightedRandomSampler(
                    weights=weights, num_samples=population_size, replacement=True
                ))
            else:
                indices = list(torch.utils.data.WeightedRandomSampler(
                    weights=weights, num_samples=population_size, replacement=True, generator=self.generator
                ))

            mating_pool = [population_mol[i] for i in indices if population_mol[i] is not None]
            mating_pool_score = [population_scores[i] for i in indices if population_mol[i] is not None]
            # print(mating_pool)
        else:
            population_scores = [s + MINIMUM for s in population_scores]
            sum_scores = sum(population_scores)
            population_probs = [p / sum_scores for p in population_scores]
            # mating_pool = np.random.choice(population_mol, p=population_probs, size=offspring_size, replace=True)
            indices = self.rng.choice(np.arange(len(population_mol)), p=population_probs, size=population_size,
                                       replace=True)
            mating_pool = [population_mol[i] for i in indices if population_mol[i] is not None]
            mating_pool_score = [population_scores[i] for i in indices if population_mol[i] is not None]

        return mating_pool, mating_pool_score

    def reproduce(self, mating_pool):
        """
        Args:
            mating_pool: list of RDKit Mol
        Returns:
        """
        if len(mating_pool) == 0:
            return None
        parent_a = self.rng.choice(mating_pool)
        parent_b = self.rng.choice(mating_pool)
        new_child = crossover(parent_a, parent_b)
        if new_child is not None:
            new_child = mutate(new_child, self.mutation_rate)
        return new_child

class GeneticOperatorHandler_GroupSelfies(GeneticOperatorHandler):

    def __init__(self, config, action_space, seed = -1):
        self.grammar_path = config.get("General", "grammar_path")
        self.grammar = GroupGrammar.from_file(self.grammar_path)
        self.mutation_rate = config.getfloat("Genetic Search", "mutation_rate")
        self.population_size = config.getint("Genetic Search", "population_size")
        self.crossover_rate = config.getfloat("Genetic Search", "crossover_rate", fallback=1.0)
        self.action_space = action_space

        super().__init__(config, seed)


    def query(self, query_size, mating_pool, rank_coefficient=0.01):
        """
        Query the genetic operators for new molecules.
        Args:
            query_size: int, number of molecules to generate
            mating_pool: tuple (list of group_selfies, list of rewards)
            rank_coefficient: float, coefficient for ranking
        """


        population_encodings = mating_pool[0]
        population_rewards = mating_pool[1]

        mating_pool_encodings, mating_pool_rewards = self.select_pop(population_encodings, population_rewards,
                                                                  self.population_size, rank_coefficient, replacement=True)

        ##random_injection = self.config.getint("Genetic Search", "random_injection", fallback=0)
        #random_injection = 0

        ##create some random genomes to inject into the mating pool
        #for _ in range(random_injection):
        #    g = GGS(grammar_path=self.grammar_path, rng=self.rng)
        #    n_groups = self.rng.integers(1, 10)
        #    g.create_random_genome(n_groups=n_groups, n_explicit_pops=5)
        #    mating_pool_encodings.append(g.encoding)
        #    print("Random injection genome: ", g.encoding)

        #pool = Parallel(n_jobs=self.n_processes)

        seed_sequence = np.random.SeedSequence(self.rng.integers(2 ** 32))
        worker_seeds = seed_sequence.generate_state(query_size)

        worker_with_args = partial(reproduce_gs_worker,
                                   mating_pool=mating_pool_encodings,
                                   grammar=self.grammar,
                                   max_seq_length=self.max_seq_length,
                                   action_space=self.action_space,
                                   crossover_rate=self.crossover_rate,
                                   mutation_rate=self.mutation_rate
                                   )

        pool = Parallel(n_jobs=self.n_processes)
        tasks = [delayed(worker_with_args)(seed=seed) for seed in worker_seeds]
        results = pool(tasks)

        all_offspring_data = [item for sublist in results if sublist for item in sublist]

        seen_encodings = set()

        offspring_encodings_filtered = []
        crossover_stats_filtered = []
        mutation_stats_filtered = []

        # Correctly filter for unique encodings and gather metadata
        for encoding, cross_stat, mut_stat in all_offspring_data:
            action_sequence = self.action_space.encoding_to_action_sequence(encoding)
            if len(action_sequence) > self.max_seq_length:
                print("Genetic search produced a sequence which is too long, skipping...")
                continue
            if encoding and encoding not in seen_encodings:
                offspring_encodings_filtered.append(encoding)
                crossover_stats_filtered.append(cross_stat)
                mutation_stats_filtered.append(mut_stat)
                seen_encodings.add(encoding)

        gc.collect()

        offspring_encodings_filtered = np.array(offspring_encodings_filtered)
        crossover_stats_filtered = np.array(crossover_stats_filtered)
        mutation_stats_filtered = np.array(mutation_stats_filtered)
        return offspring_encodings_filtered, crossover_stats_filtered, mutation_stats_filtered, mating_pool_encodings, mating_pool_rewards


def reproduce_gs_worker(mating_pool,
                    grammar,
                    max_seq_length,
                    action_space,
                    crossover_rate,
                    mutation_rate,
                    seed):
    """
    A single-threaded, parallel-safe worker function for reproduction.

    Args:
        mating_pool: list of RDKit Mol (or encodings)
        grammar (): Group Grammae
        max_seq_length (int): Max sequence length
        action_space: The action space object (must be pickleable)
        crossover_rate (float): Probability of crossover
        mutation_rate (float): Probability of mutation
        seed: A seed for the local RNG (e.g., from numpy.random.SeedSequence)

    Returns:
        A list of (encoding, crossover_stat, mutation_stat) tuples, or None.
    """

    rng = np.random.default_rng(seed)

    def mutation(parent):
        """
        Tries all available mutations on a copy of the parent and
        randomly returns one of the valid, successful mutations.

        Args:
            parent_encoding: GGS enconding

        Returns:
            (mutated_encoding, which_mutation): A tuple containing the
                resulting genome encoding and the integer type of the
                mutation that was applied.
            (None, -1): If all mutation types failed.
        """


        mutation_method_names = [
            "group_mutation",  # 0
            "bond_mutation",  # 1
            "anchor_pos_mutation",  # 2
            "insert_group_mutation",  # 3
            "insert_start_end_group_mutation",  # 4
            "truncate_mutation",  # 5
            "anchor_group_mutation",  # 6
            "insert_branch_mutation" #7:
        ]

        valid_mutations = []


        mutation_indices = rng.permutation(len(mutation_method_names))

        for which_mutation in mutation_indices:
            method_name = mutation_method_names[which_mutation]

            try:
                mutation_function = getattr(parent, method_name)
                mutated_genome = mutation_function(return_new_object=True)
                action_seq = action_space.encoding_to_action_sequence(mutated_genome.encoding)
                if len(action_seq) > max_seq_length:
                    continue
                valid_mutations.append((mutated_genome.encoding, which_mutation))

            except (ValueError, AssertionError) as e:
                continue
            except Exception as e:
                print(f"An unexpected error occurred during {method_name} ({which_mutation}): {e}")

        if not valid_mutations:
            return None, -1
        else:
            chosen_mutation = rng.choice(valid_mutations)
            return chosen_mutation

    def crossover_func(parent_a, parent_b):
        crossover = True
        try:
            g = GGS(grammar=grammar, rng=rng)

            offspring_a, offspring_b = g.single_point_crossover(parent_a, parent_b, return_new_object=True)

            return crossover, offspring_a, offspring_b

        except ValueError as e:
            crossover = False
            print(f"Error in crossover: {e}")
            return crossover, None, None
        except KeyError as e:
            crossover = False
            print(f"Error in crossover: {e}")
            return crossover, None, None

    parent_a, parent_b = rng.choice(mating_pool, size=2, replace=True)

    try:
        parent_a = GGS(parent_a, grammar, rng)
        parent_b = GGS(parent_b, grammar, rng)
    except ValueError as e:
        print(f"Error creating GGS from group_selfies: {e}")
        return None
    except KeyError as e:
        print(f"Error creating GGS from group_selfies: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error creating GGS from group_selfies: {e}")
        return None


    if len(mating_pool) == 0:
        return None

    #take care of crossover
    crossover = False
    offspring_a = parent_a
    offspring_b = parent_b
    rand_float = rng.random()
    if rand_float < crossover_rate:
        retry = 5
        while retry > 0:
            crossover, offspring_a_, offspring_b_ = crossover_func(parent_a, parent_b)


            if crossover:
                action_seq_a = action_space.encoding_to_action_sequence(offspring_a_.encoding)
                action_seq_b = action_space.encoding_to_action_sequence(offspring_b_.encoding)
                valid = True
                if len(action_seq_a) > max_seq_length or len(action_seq_b) > max_seq_length:
                    valid = False
                if valid:
                    offspring_a = offspring_a_
                    offspring_b = offspring_b_
                    crossover = True
                    break
            retry -= 1
            parent_a = rng.choice(mating_pool)
            parent_b = rng.choice(mating_pool)

            try:
                parent_a = GGS(parent_a, grammar, rng)
                parent_b = GGS(parent_b, grammar, rng)
            except ValueError as e:
                print(f"Error creating GGS from group_selfies: {e}")
                return None
            except KeyError as e:
                print(f"Error creating GGS from group_selfies: {e}")
                return None
            except Exception as e:
                print(f"Unexpected error creating GGS from group_selfies: {e}")
                return None


    return_list = []
    # Process parent A
    mutation_stat_a = -1
    crossover_stat_a = int(crossover)
    if rng.random() < mutation_rate or crossover == False:
        mutated_a, which_mutation = mutation(offspring_a)
        mutation_stat_a = which_mutation
        if mutated_a:
            return_list.append((mutated_a, crossover_stat_a, mutation_stat_a))
    else:
        return_list.append((offspring_a.encoding, crossover_stat_a, mutation_stat_a))

    # Process parent B
    mutation_stat_b = -1
    crossover_stat_b = int(crossover)
    if rng.random() < mutation_rate or crossover == False:
        mutated_b, which_mutation = mutation(offspring_b)
        mutation_stat_b = which_mutation
        if mutated_b:
            return_list.append((mutated_b, crossover_stat_b, mutation_stat_b))
    else:
        return_list.append((offspring_b.encoding, crossover_stat_b, mutation_stat_b))
    return return_list

