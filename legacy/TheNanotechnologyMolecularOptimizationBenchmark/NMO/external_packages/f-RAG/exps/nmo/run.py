# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for f-RAG. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import sys
sys.path.append('../..')

import re
import random
import argparse
import yaml
import pandas as pd
import numpy as np
import safe as sf
from rdkit import Chem
from rdkit.Chem import AllChem, QED, RDConfig
import ga.crossover as co
from ga.ga import reproduce
from fusion.sample import SAFEFusionDesign
from fusion.slicer import MolSlicer
from easydict import EasyDict

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class f_RAG():
    def __init__(self, base_args):
        super().__init__()
        self.args = EasyDict(yaml.safe_load(open('hparams.yaml')))
        self.args.oracle_name = base_args.oracle_name
        self.args.num_mols = base_args.num_mols
        self.args.seed = base_args.seed
        
        #self.predictor = DockingVina(self.args.oracle_name)
        from NMO import Oracle_Handler_Smiles
        #make path interchangabl
        self.predictor = Oracle_Handler_Smiles("config_ph_transp.ini")
        self._cache = {}
        self.oracle_calls = 0
        self.designer = SAFEFusionDesign.load_default()
        print(f"Loading designer from {self.args.injection_model_path}")
        self.designer.load_fuser(self.args.injection_model_path)
        
        self.slicer = MolSlicer(shortest_linker=True)
        self.set_initial_population()
        co.MIN_SIZE, co.MAX_SIZE = self.args.min_mol_size, self.args.max_mol_size

        self.fname = f'results/{self.args.oracle_name}_{self.args.seed}.csv'
        if not os.path.exists(os.path.dirname(self.fname)):
            os.mkdir(os.path.dirname(self.fname))        

    
    def reward(self, smiles_list):
        #mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        print(f"fitness for {smiles_list}", flush=True)
        fitness, rewards, exceeded = self.predictor.get_fitness(smiles_list)
        print(f"is {fitness}", flush=True)
        return (fitness,), fitness


        # return (), rv
    
    def attach(self, frag1, frag2):
        rxn = AllChem.ReactionFromSmarts('[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]')
        mols = rxn.RunReactants((Chem.MolFromSmiles(frag1), Chem.MolFromSmiles(frag2)))
        idx = np.random.randint(len(mols))
        return Chem.MolToSmiles(mols[idx][0])

    def fragmentize(self, smiles):
        try:
            frags = set()
            for safe_frag in self.slicer(smiles):
                if safe_frag is None:
                    continue
                smiles_frag = sf.decode(Chem.MolToSmiles(safe_frag), remove_dummies=False)
                smiles_frag = re.sub(r'\[\d+\*\]', '[1*]', smiles_frag)
                if smiles_frag.count('*') in {1, 2}:
                    frags.add(smiles_frag)
            frags = [frag for frag in frags
                     if self.args.min_frag_size <= Chem.MolFromSmiles(frag).GetNumAtoms() <= self.args.max_frag_size]
            return frags
        except KeyboardInterrupt:
            quit()
        except:
            return None
    
    def set_initial_population(self):
        #
        df = pd.read_csv(f'../../vocab/{self.args.oracle_name}.csv')
        df = pd.read_csv(f'/home/atuin/b296ee/b296ee10/f-RAG/vocab/nmo.csv')
        df = df[df['size'] >= self.args.min_frag_size]
        df = df[df['size'] <= self.args.max_frag_size]
        print("initial pop", df)
        
        self.mol_population = []                                # list of (prop, mol)
        self.arm_population, self.linker_population = [], []    # list of (prop, frag)
        for prop, frag in zip(df['score'], df['frag']):
            if frag.count('*') == 1:
                self.arm_population.append((prop, frag))
            else:
                self.linker_population.append((prop, frag))
            if (len(self.arm_population) >= self.args.frag_population_size and
                len(self.linker_population) >= self.args.frag_population_size):
                break
        self.arm_population = self.arm_population[:self.args.frag_population_size]
        self.linker_population = self.linker_population[:self.args.frag_population_size]
        print("arm pop", self.arm_population)
        print("linker_population", self.linker_population)
        
    def update_population(self, prop_list, smiles_list):
        self.mol_population += list(set(zip(prop_list, smiles_list)))
        self.mol_population.sort(reverse=True)
        self.mol_population = self.mol_population[:self.args.mol_population_size]

        arms = {frag for prop, frag in self.arm_population}
        linkers = {frag for prop, frag in self.linker_population}
        for prop, smiles in zip(prop_list, smiles_list):
            frags = self.fragmentize(smiles)
            if frags is not None:
                for frag in frags:
                    if frag.count('*') == 1 and frag not in arms:
                        self.arm_population.append((prop, frag))
                    elif frag.count('*') == 2 and frag not in linkers:
                        self.linker_population.append((prop, frag))
        
        self.arm_population.sort(reverse=True)
        self.linker_population.sort(reverse=True)
        self.arm_population = self.arm_population[:self.args.frag_population_size]
        self.linker_population = self.linker_population[:self.args.frag_population_size]

    def generate(self):
        for i in range(1000):
            try:
                if random.random() < 0.5:   # arm + arm
                    frag1, frag2 = random.sample([frag for prop, frag in self.arm_population], 2)
                    # retrieval population <- linker population
                    self.designer.frags = [frag for prop, frag in self.linker_population]
                    smiles = self.designer.linker_generation(frag1, frag2,
                                                             n_samples_per_trial=1,
                                                             random_seed=self.args.seed)[0]
                else:                       # arm + linker
                    frag1 = random.choice([frag for prop, frag in self.arm_population])
                    frag2 = random.choice([frag for prop, frag in self.linker_population])
                    frag = re.sub(r'\[1\*\]', '[*]', self.attach(frag1, frag2))
                    # retrieval population <- arm population
                    self.designer.frags = [frag for prop, frag in self.arm_population]
                    smiles = self.designer.motif_extension(frag,
                                                           n_samples_per_trial=1,
                                                           random_seed=self.args.seed)[0]
                    smiles = sorted(smiles.split('.'), key=len)[-1]     # the largest
                smiles = sf.decode(smiles)
                if self.args.min_mol_size <= Chem.MolFromSmiles(smiles).GetNumAtoms() <= self.args.max_mol_size:
                    return smiles
            except KeyboardInterrupt:
                quit()
            except Exception as e:
                print(f"Exception {e}")
                continue
    
    def record(self, smiles_list, prop_list, rews):
        with open(self.fname, 'a') as f:
            for i in range(len(smiles_list)):
                str = f'{smiles_list[i]},'
                for rew in rews: str += f'{rew[i]},'
                str += f'{prop_list[i]}\n'
                f.write(str)

    def run(self):
        num_generated = 0
        while True:
            # SAFE-GPT generation
            safe_smiles_list = [self.generate() for _ in range(self.args.num_safe)]
            safe_rews, safe_prop_list = self.reward(safe_smiles_list)
            self.update_population(safe_prop_list, safe_smiles_list)
            self.record(safe_smiles_list, safe_prop_list, safe_rews)
            num_generated += len(safe_smiles_list)

            # GA generation
            if len(self.mol_population) == self.args.mol_population_size:
                ga_smiles_list = [reproduce(self.mol_population, self.args.mutation_rate)
                                  for _ in range(self.args.num_ga)]
                ga_rews, ga_prop_list = self.reward(ga_smiles_list)
                self.update_population(ga_prop_list, ga_smiles_list)
                self.record(ga_smiles_list, ga_prop_list, ga_rews)
                num_generated += len(ga_smiles_list)

            if num_generated >= self.args.num_mols:
                break
            if self.predictor.oracle_calls >= self.predictor.max_oracle_calls:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--oracle_name', type=str, default='nmo',
                        choices=['nmo'])
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-n', '--num_mols', type=int, default=3000)
    base_args = parser.parse_args()

    f_RAG(base_args).run()
