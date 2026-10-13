"""GGS-space genetic operators for mol_ga.

Drop-in replacement for graph_ga_blended_generation when the population is
stored as GGS encoding strings.  The signature of ggs_blended_generation
matches mol_ga's offspring_gen_func interface; bind `grammar` with
functools.partial before passing to default_ga:

    offspring_fn = functools.partial(ggs_blended_generation, grammar=grammar)
    default_ga(..., offspring_gen_func=offspring_fn)

Two operator modes are available:

  use_native_operators=False (default)
      Simple string/token-level operators: swap a group token (mutate) or
      splice unit-boundary prefixes/suffixes (crossover).

  use_native_operators=True
      Graph-level operators from the GGS class itself, matching those used in
      BeyondDrugDiscovery-GGS-NMO.  Mutations available:
        group_mutation, bond_mutation, anchor_pos_mutation,
        anchor_group_mutation, insert_group_mutation,
        insert_start_end_group_mutation, truncate_mutation,
        insert_branch_mutation.
      Crossover uses GGS.single_point_crossover, which yields two children per
      pair.  Pass mutation_operators to restrict to a subset.
"""

from __future__ import annotations

import random
import re
from typing import Optional

import joblib
import numpy as np

from GGS import GGS


# ---------------------------------------------------------------------------
# Native (graph-level) operator wrappers
# ---------------------------------------------------------------------------

#: All graph-level mutation method names exposed by GGS.
ALL_NATIVE_MUTATIONS: list[str] = [
    "group_mutation",
    "bond_mutation",
    "anchor_pos_mutation",
    "anchor_group_mutation",
    "insert_group_mutation",
    "insert_start_end_group_mutation",
    "truncate_mutation",
    "insert_branch_mutation",
]


def _np_rng(rng: random.Random) -> np.random.Generator:
    """Derive a numpy Generator from a random.Random instance."""
    return np.random.default_rng(rng.randint(0, 2**31))


def _native_mutate(
    encoding: str,
    grammar,
    rng: random.Random,
    mutation_operators: list[str] = ALL_NATIVE_MUTATIONS,
) -> Optional[str]:
    """Apply one randomly chosen graph-level GGS mutation.

    Tries every operator in *mutation_operators* (shuffled) and returns the
    first encoding that passes validation.  Returns None if all fail.
    """
    np_rng = _np_rng(rng)
    try:
        ggs = GGS(encoding, grammar=grammar, rng=np_rng)
    except Exception:
        return None

    ops = list(mutation_operators)
    rng.shuffle(ops)
    for op_name in ops:
        try:
            result = getattr(ggs, op_name)()  # returns (encoding, graph)
            new_enc = result[0]
            if _is_valid(new_enc, grammar):
                return new_enc
        except Exception:
            continue
    return None


def _native_crossover(
    enc_a: str, enc_b: str, grammar, rng: random.Random
) -> list[str]:
    """Apply GGS.single_point_crossover to produce up to two children.

    Returns a (possibly empty) list of valid encoding strings.
    """
    np_rng = _np_rng(rng)
    try:
        g1 = GGS(enc_a, grammar=grammar, rng=np_rng)
        g2 = GGS(enc_b, grammar=grammar, rng=np_rng)
        (enc1, _), (enc2, _) = g1.single_point_crossover(g1, g2)
        return [e for e in (enc1, enc2) if e is not None and _is_valid(e, grammar)]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def _tokenize(encoding: str) -> list[str]:
    """Return every [...] token in the encoding, in order."""
    return re.findall(r'\[.*?\]', encoding)


def _split_into_units(encoding: str) -> list[list[str]]:
    """Split a GGS encoding into token units.

    A unit starts with a group token (contains ':') or [pop], followed by
    zero or more shift tokens.  Mirrors GGS.split_group_selfies logic.
    """
    units: list[list[str]] = []
    current: list[str] = []
    for tok in _tokenize(encoding):
        if ':' in tok or tok == '[pop]':
            if current:
                units.append(current)
            current = [tok]
        else:
            current.append(tok)
    if current:
        units.append(current)
    return units


def _ap_count(grammar) -> dict[str, int]:
    """Return {group_name: n_attachment_points} for every grammar group."""
    return {
        name: len(grammar.vocab[name].attachment_points)
        for name in grammar.vocab
    }


def _groups_by_ap(grammar) -> dict[int, list[str]]:
    """Return {n_attachment_points: [group_names]} for every grammar group."""
    result: dict[int, list[str]] = {}
    for name, n in _ap_count(grammar).items():
        if n > 0:
            result.setdefault(n, []).append(name)
    return result


def _is_valid(encoding: str, grammar) -> bool:
    try:
        GGS(encoding, grammar=grammar)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Mutation: swap one group token for another with the same AP count
# ---------------------------------------------------------------------------

def mutate_ggs(encoding: str, grammar, rng: random.Random) -> Optional[str]:
    """Replace a random group fragment with another of equal attachment-point count.

    Keeps the start-attachment-point index unchanged (safe because the
    replacement group has the same AP count).  Returns None if no valid
    mutant is found after trying every group token.
    """
    raw = _tokenize(encoding)
    group_indices = [i for i, t in enumerate(raw) if ':' in t]
    if not group_indices:
        return None

    ap_of = _ap_count(grammar)
    by_ap = _groups_by_ap(grammar)

    rng.shuffle(group_indices)
    for idx in group_indices:
        m = re.match(r'\[:(\d+)(.*)\]', raw[idx])
        if not m:
            continue
        start_ap, group_name = m.group(1), m.group(2)
        if group_name not in ap_of:
            continue

        candidates = [g for g in by_ap.get(ap_of[group_name], []) if g != group_name]
        if not candidates:
            continue

        new_raw = raw.copy()
        new_raw[idx] = f'[:{start_ap}{rng.choice(candidates)}]'
        new_enc = ''.join(new_raw)
        if _is_valid(new_enc, grammar):
            return new_enc

    return None


# ---------------------------------------------------------------------------
# Crossover: splice two encodings at unit boundaries
# ---------------------------------------------------------------------------

def crossover_ggs(
    enc_a: str, enc_b: str, grammar, rng: random.Random
) -> Optional[str]:
    """Combine a prefix of enc_a with a suffix of enc_b at unit boundaries.

    Tries up to 10 random split points; discards splices that fail
    GGS.encoding_to_graph() validation.  Returns None if all attempts fail.
    """
    units_a = _split_into_units(enc_a)
    units_b = _split_into_units(enc_b)
    if len(units_a) < 2 or len(units_b) < 2:
        return None

    for _ in range(10):
        split_a = rng.randint(1, len(units_a) - 1)
        split_b = rng.randint(1, len(units_b) - 1)
        new_enc = ''.join(
            ''.join(u) for u in units_a[:split_a] + units_b[split_b:]
        )
        if _is_valid(new_enc, grammar):
            return new_enc

    return None


# ---------------------------------------------------------------------------
# Random fallback
# ---------------------------------------------------------------------------

def _random_genome(grammar, rng: random.Random, n_groups: int, n_pops: int) -> Optional[str]:
    try:
        np_rng = np.random.default_rng(rng.randint(0, 2**31))
        g = GGS(grammar=grammar, rng=np_rng)
        return g.create_random_genome(n_groups, n_pops)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main offspring generator
# ---------------------------------------------------------------------------

def ggs_blended_generation(
    samples: list[str],
    n_candidates: int,
    rng: random.Random,
    parallel: Optional[joblib.Parallel],
    *,
    grammar,
    frac_mutate: float = 0.15,
    n_groups_random: int = 5,
    n_pops_random: int = 2,
    use_native_operators: bool = False,
    mutation_operators: list[str] = ALL_NATIVE_MUTATIONS,
) -> set[str]:
    """Generate GGS offspring by mutation and crossover.

    Mirrors graph_ga_blended_generation's structure and has the same
    4-positional-argument signature once grammar is bound via functools.partial.

    Parameters
    ----------
    use_native_operators:
        If False (default), use the simple string/token-level operators
        (mutate_ggs, crossover_ggs).  If True, use the graph-level operators
        from the GGS class: a randomly chosen mutation from *mutation_operators*
        and GGS.single_point_crossover (which yields two children per pair).
    mutation_operators:
        Which graph-level mutations to use when use_native_operators=True.
        Defaults to ALL_NATIVE_MUTATIONS.  Ignored when use_native_operators=False.

    If mutation + crossover yield fewer than n_candidates/2 valid offspring,
    random genomes fill the gap (bounded number of attempts).
    """
    samples_mutate: list[str] = []
    samples_crossover: list[str] = []
    for s in samples:
        if rng.random() < frac_mutate:
            samples_mutate.append(s)
        else:
            samples_crossover.append(s)
    samples_mutate = samples_mutate[: int(n_candidates * frac_mutate + 1)]

    # ------------------------------------------------------------------ #
    # Mutations
    # ------------------------------------------------------------------ #
    if use_native_operators:
        if parallel:
            offspring: list[Optional[str]] = parallel(
                joblib.delayed(_native_mutate)(s, grammar, rng, mutation_operators)
                for s in samples_mutate
            )
        else:
            offspring = [
                _native_mutate(s, grammar, rng, mutation_operators)
                for s in samples_mutate
            ]
    else:
        if parallel:
            offspring = parallel(
                joblib.delayed(mutate_ggs)(s, grammar, rng) for s in samples_mutate
            )
        else:
            offspring = [mutate_ggs(s, grammar, rng) for s in samples_mutate]

    # ------------------------------------------------------------------ #
    # Crossovers
    # ------------------------------------------------------------------ #
    n_crossover = n_candidates - len(offspring)
    pairs = list(samples_crossover)
    rng.shuffle(pairs)

    if use_native_operators:
        # single_point_crossover yields two children per pair — collect both
        if parallel:
            child_lists: list[list[str]] = parallel(
                joblib.delayed(_native_crossover)(s1, s2, grammar, rng)
                for s1, s2 in zip(samples_crossover[:n_crossover], pairs)
            )
        else:
            child_lists = [
                _native_crossover(s1, s2, grammar, rng)
                for s1, s2 in zip(samples_crossover[:n_crossover], pairs)
            ]
        for children in child_lists:
            offspring.extend(children)
    else:
        if parallel:
            offspring += parallel(
                joblib.delayed(crossover_ggs)(s1, s2, grammar, rng)
                for s1, s2 in zip(samples_crossover[:n_crossover], pairs)
            )
        else:
            offspring += [
                crossover_ggs(s1, s2, grammar, rng)
                for s1, s2 in zip(samples_crossover[:n_crossover], pairs)
            ]

    result: set[str] = {o for o in offspring if o is not None and isinstance(o, str)}

    # Random fill if operators produced too few valid offspring
    max_fill = n_candidates * 3
    attempts = 0
    while len(result) < n_candidates // 2 and attempts < max_fill:
        attempts += 1
        enc = _random_genome(grammar, rng, n_groups_random, n_pops_random)
        if enc is not None:
            result.add(enc)

    return result
