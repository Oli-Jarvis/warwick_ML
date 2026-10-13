"""Maximize NMO oracle fitness using mol_ga's genetic algorithm.

Analogous to maximize_toy_oracles.py, but driven by Oracle_Handler_Smiles from
the NMO benchmark.  Termination is controlled purely by the oracle budget set in
the config (max_oracle_calls), not by a fixed generation count.

Oracle-call accounting
----------------------
mol_ga wraps scoring_fn in CachedBatchFunction, so only *novel* SMILES ever
reach Oracle_Handler_Smiles.get_fitness().  The NMO oracle's internal counter
therefore tracks unique evaluations only — no double-counting from the cache.

Budget breakdown (approximate upper bound):
    start_population_size             <- consumed before generation 0
    + max_generations * offspring_size  <- per-generation new calls (cache trims this)
Set max_oracle_calls in the config to cover however much of this you want.
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import joblib
import numpy as np
from typing import Optional

from mol_ga.preconfigured_gas import default_ga


class BudgetExceeded(Exception):
    pass


class NMOScoringFn:
    """Adapts Oracle_Handler_Smiles to mol_ga's scoring interface.

    mol_ga expects:  list[str] -> list[float]
    NMO returns:     (np.ndarray, dict, bool)

    Also accumulates every (smiles, score) pair evaluated so results are
    recoverable even when the GA is interrupted by BudgetExceeded.
    """

    def __init__(self, oracle, max_tokens: Optional[int] = 15):
        self.oracle = oracle
        self.max_tokens = max_tokens
        self._all_scored: list[tuple[str, float]] = []

    def __call__(self, smiles_list: list[str]) -> list[float]:
        from gss_operators import _tokenize

        if self.max_tokens is not None:
            oracle_input = [
                s if len(_tokenize(s)) <= self.max_tokens else ""
                for s in smiles_list
            ]
        else:
            oracle_input = smiles_list

        fitness, _, exceeded = self.oracle.get_fitness(oracle_input)
        scores = fitness.tolist()
        self._all_scored.extend(zip(smiles_list, scores))
        if exceeded:
            raise BudgetExceeded()
        return scores

    @property
    def all_scored(self) -> dict[str, float]:
        """All evaluated (smiles -> best score) pairs, deduplicated."""
        result: dict[str, float] = {}
        for smiles, score in self._all_scored:
            if smiles not in result or score > result[smiles]:
                result[smiles] = score
        return result


MODES = ["smiles", "ggs", "ggs_native"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "config_nmo.ini"),
        help="Path to NMO oracle .ini config file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="smiles",
        choices=MODES,
        help=(
            "GA representation and operator set:\n"
            "  smiles     — SMILES population, graph-GA operators (default)\n"
            "  ggs        — GGS population, simple string/token operators\n"
            "  ggs_native — GGS population, graph-level operators from GGS class"
        ),
    )
    parser.add_argument(
        "--offspring_size",
        type=int,
        default=200,
        help="Number of offspring per generation",
    )
    parser.add_argument(
        "--start_population_size",
        type=int,
        default=1000,
        help="Size of the initial population scored before generation 0",
    )
    args = parser.parse_args()
    logging.info(f"Arguments: {args}")

    # ------------------------------------------------------------------ #
    # Build starting population and offspring function based on --mode
    # ------------------------------------------------------------------ #
    import configparser
    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    if args.mode == "smiles":
        from mol_ga.mol_libraries import random_zinc as _rand_pop
        starting_population = _rand_pop(args.start_population_size)
        offspring_fn = None  # default_ga uses its built-in graph-GA operator

    else:  # ggs or ggs_native
        from group_selfies import GroupGrammar
        import functools
        from gss_population import random_ggs
        from gss_operators import ggs_blended_generation

        grammar = GroupGrammar.from_file(cfg["General"]["grammar_path"])
        starting_population = random_ggs(grammar, size=args.start_population_size)
        offspring_fn = functools.partial(
            ggs_blended_generation,
            grammar=grammar,
            use_native_operators=(args.mode == "ggs_native"),
        )

    # ------------------------------------------------------------------ #
    # Oracle — class is determined entirely by the config file
    # ------------------------------------------------------------------ #
    logging.info("Initialising NMO oracle...")
    if args.mode == "smiles":
        from NMO import Oracle_Handler_Smiles
        oracle = Oracle_Handler_Smiles(args.config)
    else:
        from NMO import Oracle_Handler_GGS
        oracle = Oracle_Handler_GGS(args.config)

    scoring_fn = NMOScoringFn(oracle)

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #
    ga_kwargs = dict(
        starting_population_smiles=starting_population,
        scoring_function=scoring_fn,
        max_generations=100_000,
        offspring_size=args.offspring_size,
    )
    if offspring_fn is not None:
        ga_kwargs["offspring_gen_func"] = offspring_fn

    start_time = time.monotonic()
    try:
        with joblib.Parallel(n_jobs=1) as parallel:
            output = default_ga(parallel=parallel, **ga_kwargs)
        all_scored = output.scoring_func_evals
    except BudgetExceeded:
        logging.info("Oracle budget exhausted — stopping GA.")
        all_scored = scoring_fn.all_scored
    end_time = time.monotonic()

    top_scores = sorted(all_scored.values(), reverse=True)
    top_100 = top_scores[:100]

    print(f"\nTotal unique molecules evaluated: {len(all_scored)}")
    print(f"NMO oracle call counter:          {oracle.oracle_calls}")
    print(f"Time elapsed: {end_time - start_time:.2f} seconds")
    print("\nTop 25 scores:")
    print(top_100[:25])

    if len(top_100) == 100:
        guacamol = (top_100[0] + np.mean(top_100[:10]) + np.mean(top_100[:100])) / 3
        print(f"\nGuacamol top-1/10/100 average: {guacamol:.3f}")
