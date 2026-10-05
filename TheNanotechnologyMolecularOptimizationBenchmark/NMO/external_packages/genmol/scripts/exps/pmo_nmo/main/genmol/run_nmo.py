# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import random
import torch
import numpy as np
import yaml
import pandas as pd
from rdkit import Chem
from easydict import EasyDict
from main.optimizer import Oracle, top_auc
from main.genmol.run import GenMolOpt


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


class NMOOracle(Oracle):
    """Adapts Oracle_Handler_Smiles to match the Oracle interface.

    Oracle_Handler_Smiles.get_fitness(smiles_list) returns:
        (fitness: np.array, rewards: dict, oracle_calls_exceeded: bool)

    This class wraps that interface so the rest of GenMol's infrastructure
    (mol_buffer, logging, save_result) works unchanged.
    """

    def __init__(self, args=None):
        super().__init__(args)
        self._oracle_calls_exceeded = False

    def assign_evaluator(self, evaluator):
        self.evaluator = evaluator

    def score_smi(self, smi):
        if self._oracle_calls_exceeded:
            return 0
        if smi is None:
            return 0
        mol = Chem.MolFromSmiles(smi)
        if mol is None or len(smi) == 0:
            return 0
        smi = Chem.MolToSmiles(mol)
        if smi in self.mol_buffer:
            return self.mol_buffer[smi][0]
        fitness, rewards, exceeded = self.evaluator.get_fitness([smi])
        self._oracle_calls_exceeded = exceeded
        score = float(fitness[0])
        self.mol_buffer[smi] = [score, len(self.mol_buffer) + 1]
        return score

    @property
    def finish(self):
        return self._oracle_calls_exceeded or len(self.mol_buffer) >= self.max_oracle_calls


class NMO_Optimizer:
    """GenMol optimizer for the NMO benchmark.

    Mirrors the structure of GenMol_Optimizer / BaseOptimizer but avoids
    TDC-specific initialisation (tdc.Oracle, tdc.Evaluator, tdc.MolFilter).
    """

    def __init__(self, args=None):
        self.model_name = 'GenMol_NMO'
        self.args = args
        self.n_jobs = args.n_jobs
        self.oracle = NMOOracle(args=self.args)

    # ------------------------------------------------------------------
    # Forwarding helpers that mirror BaseOptimizer
    # ------------------------------------------------------------------

    @property
    def mol_buffer(self):
        return self.oracle.mol_buffer

    @property
    def finish(self):
        return self.oracle.finish

    def sort_buffer(self):
        self.oracle.sort_buffer()

    def log_intermediate(self, mols=None, scores=None, finish=False):
        self.oracle.log_intermediate(mols=mols, scores=scores, finish=finish)

    def save_result(self, suffix=None):
        print("Saving molecules...")
        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, 'results.yaml')
        else:
            output_file_path = os.path.join(self.args.output_dir, 'results_' + suffix + '.yaml')
        self.sort_buffer()
        with open(output_file_path, 'w') as f:
            yaml.dump(self.mol_buffer, f, sort_keys=False)

    def reset(self):
        del self.oracle
        self.oracle = NMOOracle(args=self.args)

    # ------------------------------------------------------------------
    # Core optimization logic
    # ------------------------------------------------------------------

    def _optimize(self, oracle, config):
        self.oracle.assign_evaluator(oracle)
        config = EasyDict(config)
        config.seed = self.args.seed
        config.oracle_name = self.args.task_name
        NMOOpt(config, self.oracle).run()

    def optimize(self, oracle, config, seed=0):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        self.seed = seed
        task_label = f'{self.model_name}_{self.args.task_name}_{seed}'
        self.oracle.task_label = task_label
        self._optimize(oracle, config)
        self.save_result(task_label)
        self.reset()


class NMOOpt(GenMolOpt):
    """GenMolOpt variant that uses the NMO uniform vocabulary.

    The only difference from GenMolOpt is that the vocabulary is always
    loaded from vocab/nmo_uniform.csv instead of an oracle-specific CSV.
    All fragment assembly, generation, and population update logic is
    inherited unchanged.
    """

    def set_initial_population(self):
        vocab_path = os.path.join(ROOT_DIR, 'vocab/nmo_uniform.csv')
        df = pd.read_csv(vocab_path)
        df = df.iloc[:self.args.population_size]
        self.population = list(zip(df['score'], df['frag']))
