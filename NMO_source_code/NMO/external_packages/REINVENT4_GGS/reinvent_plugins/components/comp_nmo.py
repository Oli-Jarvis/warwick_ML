"""NMO Benchmark Oracles for REINVENT4 — Group SELFIES index encoding"""
from __future__ import annotations

__all__ = ["nmo"]

from dataclasses import dataclass
from typing import List
import logging

import numpy as np

from .component_results import ComponentResults
from .add_tag import add_tag
from NMO import Oracle_Handler_GGS
from reinvent.action_space import Action_Space_GroupSelfies

logger = logging.getLogger("reinvent")


@add_tag("__parameters")
@dataclass
class Parameters:
    config_path: List[str]


@add_tag("__component")
class nmo:
    def __init__(self, params: Parameters):
        grammar_path = "../tests/GS_complex_grammar_without_S.txt"
        self.action_space = Action_Space_GroupSelfies.from_grammar_path(
            grammar_path
        )
        self.oracles = []
        for config_path in params.config_path:
            logger.info(f"Loading NMO config: {config_path}")
            self.oracles.append(Oracle_Handler_GGS(config_path))

    def __call__(self, sequences: List[str]) -> ComponentResults:
        cleaned = self.normalize(sequences)
        scores = []
        for oracle in self.oracles:
            print(f"Fitness for {cleaned}")
            fitness, rewards, exceeded = oracle.get_fitness(cleaned)
            if exceeded:
                logger.warning("NMO oracle call budget exceeded")
            result = np.array([
                f if gsf != "INVALID" else np.nan
                for f, gsf in zip(fitness, cleaned)
            ])
            scores.append(result)
        return ComponentResults(scores)

    def normalize(self, sequences: List[str]) -> List[str]:
        cleaned = []
        for seq in sequences:
            try:
                action_sequence = [int(x) for x in seq.strip().split()]
                if not self.action_space.has_end_token(action_sequence):
                    cleaned.append("INVALID")
                    continue
                group_selfies = self.action_space.action_sequence_to_encoding(
                    action_sequence
                )
                if not group_selfies:
                    cleaned.append("INVALID")
                    continue
                cleaned.append(group_selfies)
            except Exception as e:
                logger.debug(f"Failed to decode sequence '{seq}': {e}")
                cleaned.append("INVALID")
        n_invalid = cleaned.count("INVALID")
        if n_invalid:
            logger.info(f"Decoded {len(cleaned) - n_invalid}/{len(cleaned)} sequences successfully")
        return cleaned