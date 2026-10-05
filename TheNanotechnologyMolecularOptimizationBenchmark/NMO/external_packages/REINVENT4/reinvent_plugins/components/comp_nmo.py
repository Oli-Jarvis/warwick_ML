"""NMO Benchmark Oracles for REINVENT4"""

from __future__ import annotations


__all__ = ["nmo"]
from dataclasses import dataclass
from typing import List
import logging

import numpy as np
from rdkit import Chem

from .component_results import ComponentResults
from .add_tag import add_tag

from NMO import Oracle_Handler_Smiles

logger = logging.getLogger("reinvent")


@add_tag("__parameters")
@dataclass
class Parameters:
    config_path: List[str]


@add_tag("__component")
class nmo:
    def __init__(self, params: Parameters):
        self.oracles = []
        for config_path in params.config_path:
            logger.info(f"Loading NMO config: {config_path}")
            self.oracles.append(Oracle_Handler_Smiles(config_path))

    def __call__(self, smilies: List[str]) -> ComponentResults:
        cleaned_smilies = self.normalize(smilies)
        scores = []
        for oracle in self.oracles:
            
            fitness, rewards, exceeded = oracle.get_fitness(cleaned_smilies)
            result = np.array([
                f if smi != "INVALID" else np.nan
                for f, smi in zip(fitness, cleaned_smilies)
            ])
            scores.append(result)
        
        return ComponentResults(scores)

    def normalize(self, smilies: List[str]) -> List[str]:
        cleaned = []
        for smi in smilies:
            mol = Chem.MolFromSmiles(smi)
            if not mol:
                cleaned.append("INVALID")
                continue
            for atom in mol.GetAtoms():
                atom.SetIsotope(0)
                atom.SetAtomMapNum(0)
            cleaned.append(Chem.MolToSmiles(mol))
        return cleaned