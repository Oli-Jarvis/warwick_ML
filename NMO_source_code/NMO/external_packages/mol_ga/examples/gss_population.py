"""Random GGS starting population generator for mol_ga."""

from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np

from GGS import GGS


def random_ggs(
    grammar,
    size: int,
    n_groups: int = 5,
    n_pops: int = 2,
    rng: Optional[random.Random] = None,
) -> list[str]:
    """Generate `size` random valid GGS encodings from the grammar.

    Replaces random_zinc() when running the GA in GGS-encoding space.
    Each encoding is validated by GGS.encoding_to_graph() during construction.

    Args:
        grammar:  Pre-loaded GroupGrammar object.
        size:     Number of encodings to generate.
        n_groups: Max number of fragment groups per molecule (controls size).
        n_pops:   Max explicit pop tokens per molecule (controls branching).
        rng:      stdlib random.Random used to seed per-call numpy RNGs.

    Returns:
        List of valid GGS encoding strings, may be shorter than size if
        grammar is very constrained.
    """
    rng = rng or random.Random()
    results: list[str] = []
    max_attempts = size * 20
    attempts = 0

    while len(results) < size and attempts < max_attempts:
        attempts += 1
        try:
            np_rng = np.random.default_rng(rng.randint(0, 2**31))
            g = GGS(grammar=grammar, rng=np_rng)
            encoding = g.create_random_genome(n_groups, n_pops)
            results.append(encoding)
        except Exception:
            pass

    if len(results) < size:
        logging.warning(
            f"random_ggs: generated {len(results)}/{size} valid encodings "
            f"after {max_attempts} attempts."
        )
    return results
