"""NMO Benchmark Oracles for REINVENT4 — Group SELFIES index encoding"""
from __future__ import annotations

__all__ = ["nmo_group_selfies"]

from dataclasses import dataclass, field
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
    config_path: List[str]          # mirrors original — one per oracle
    grammar_path: str               # replaces RDKit — needed for index → Group SELFIES


@add_tag("__component")
class nmo_group_selfies:
    def __init__(self, params: Parameters):
        # Replaces RDKit — translates index sequences to Group SELFIES
        self.action_space = Action_Space_GroupSelfies.from_grammar_path(
            params.grammar_path
        )

        # Mirrors original — one Oracle_Handler_GGS per config
        # Oracle_Handler_GGS takes Group SELFIES directly (vs Oracle_Handler_Smiles)
        self.oracles = []
        for config_path in params.config_path:
            logger.info(f"Loading NMO config: {config_path}")
            self.oracles.append(Oracle_Handler_GGS(config_path))

    def __call__(self, sequences: List[str]) -> ComponentResults:
        # Replaces smilies — decode index sequences to Group SELFIES
        # normalize() → decode(): same role, different encoding
        group_selfies_list = self.decode(sequences)

        scores = []
        for oracle in self.oracles:
            fitness, rewards, exceeded = oracle.get_fitness(group_selfies_list)

            if exceeded:
                logger.warning("NMO oracle call budget exceeded")

            # Mirrors original — INVALID → np.nan, valid → raw fitness
            # No clipping: REINVENT applies transform from TOML config
            result = np.array([
                f if gsf != "INVALID" else np.nan
                for f, gsf in zip(fitness, group_selfies_list)
            ])
            scores.append(result)

        return ComponentResults(scores)

    def decode(self, sequences: List[str]) -> List[str]:
        """Convert index sequences to Group SELFIES strings.

        Mirrors normalize() in the original SMILES component:
        - normalize() uses Chem.MolFromSmiles() to validate
        - decode() uses has_end_token() + action_sequence_to_encoding()

        Args:
            sequences: REINVENT outputs, e.g. ["3 0 7 2 1", ...]

        Returns:
            List of Group SELFIES strings or "INVALID"
        """
        decoded = []
        for seq in sequences:
            try:
                action_sequence = [int(x) for x in seq.strip().split()]

                # Replaces Chem.MolFromSmiles() — validates sequence structure
                if not self.action_space.has_end_token(action_sequence):
                    decoded.append("INVALID")
                    continue

                group_selfies = self.action_space.action_sequence_to_encoding(
                    action_sequence
                )

                if not group_selfies:
                    decoded.append("INVALID")
                    continue

                decoded.append(group_selfies)

            except Exception as e:
                logger.debug(f"Failed to decode sequence '{seq}': {e}")
                decoded.append("INVALID")

        n_invalid = decoded.count("INVALID")
        if n_invalid:
            logger.info(
                f"Decoded {len(decoded) - n_invalid}/{len(decoded)} sequences successfully"
            )

        return decoded