"""
test_syntax_masking.py
----------------------
Test GGS syntax masking against saved action sequences.

Usage:
    python test_syntax_masking.py --project path/to/your/project
                                  --grammar  path/to/grammar.txt
                                  --sequences path/to/sequences.csv
                                  [--verbose]

    --project   : directory containing action_space.py
                  (defaults to the directory of this script)
    --grammar   : GGS grammar .txt file
    --sequences : one action sequence per row, comma or space separated ints
    --verbose   : print full state trace for every sequence

Example sequence file:
    0,5,12,8,3,91
    0,5,13,8,3,91
"""

import argparse
import sys
import os
import numpy as np


# ── state constants (must match model.py) ────────────────────────────────────
STATE_START        = 0
STATE_SI           = 1
STATE_GROUP        = 2
STATE_SO           = 3
STATE_POP          = 4
STATE_SO_AFTER_POP = 5

STATE_NAMES = {
    STATE_START:        "START",
    STATE_SI:           "SI (coupling_in)",
    STATE_GROUP:        "GROUP",
    STATE_SO:           "SO (shift_out)",
    STATE_POP:          "POP",
    STATE_SO_AFTER_POP: "SO_AFTER_POP",
}


# ── helpers ───────────────────────────────────────────────────────────────────
def get_token_state(token_idx, action_space):
    """Map a token index to its state category."""
    n_g  = action_space.n_groups
    n_si = action_space.n_si
    n_so = action_space.n_so

    if token_idx < n_g:
        return STATE_GROUP
    elif token_idx < n_g + n_si:
        return STATE_SI
    elif token_idx < n_g + n_si + n_so:
        return STATE_SO
    else:
        name = action_space.action_space[token_idx]
        if name == "pop":   return STATE_POP
        if name == "Start": return STATE_START
        return None  # End token


def allowed_types_str(state, action_space):
    """Human-readable summary of which token types are allowed in a state."""
    mask = action_space.allowed_mask[state]
    n_g, n_si, n_so = action_space.n_groups, action_space.n_si, action_space.n_so
    parts = []
    if mask[:n_g].any():                    parts.append("GROUP")
    if mask[n_g:n_g+n_si].any():           parts.append("SI")
    if mask[n_g+n_si:n_g+n_si+n_so].any(): parts.append("SO")
    if mask[action_space.idx_pop].item():   parts.append("POP")
    if mask[action_space.idx_end].item():   parts.append("END")
    return ", ".join(parts) if parts else "NOTHING"


# ── sequence loader ───────────────────────────────────────────────────────────
def load_sequences(filepath):
    """
    Load action sequences from the experience memory CSV.
    Format: semicolon-delimited, first column is the action sequence stored
    as a numpy array string e.g. [66  1 78 66 41 78 90]
    Third column (index 2) is the fitness score — rows with fitness <= 0 are skipped.
    Header row is skipped automatically.
    """
    sequences = []
    skipped_fitness = 0
    with open(filepath, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            cols = line.split(";")
            first_col = cols[0].strip()
            if not first_col.startswith("["):
                print(f"  Skipping header line {line_num}: '{line[:60]}'")
                continue
            # fitness is the third column (index 2)
            try:
                fitness = float(cols[2].strip())
            except (IndexError, ValueError):
                fitness = 0.0
            if fitness <= 0:
                skipped_fitness += 1
                continue
            inner = first_col.strip("[]")
            try:
                seq = np.array([int(t) for t in inner.split()], dtype=np.int64)
                sequences.append(seq)
            except ValueError as e:
                print(f"  Warning: could not parse line {line_num}: {e}")
    print(f"  Skipped {skipped_fitness} sequences with fitness <= 0")
    sequences = [[71, 64, 89, 86, 66,  8, 89, 84, 67, 41, 89, 84, 79, 66, 15, 89, 82, 69, 37, 78, 69, 56, 80, 66, 4, 79, 67, 53, 78, 68],]
    return sequences


# ── core checker ─────────────────────────────────────────────────────────────
def check_sequence(action_sequence, action_space):
    """
    Walk every token and verify the transition is allowed by the masking state machine.

    Returns:
        valid        : bool
        violations   : list of dicts
        states_trace : list of (token_idx, token_name, state_before, state_after)
    """
    allow_matrix  = action_space.allowed_mask
    end_idx       = action_space.reversed_action_space["End"]
    start_idx     = action_space.reversed_action_space["Start"]

    violations    = []
    states_trace  = []
    current_state = STATE_START

    for step, raw in enumerate(action_sequence):
        token_idx  = int(raw)
        token_name = action_space.action_space.get(token_idx, str(token_idx))

        if token_idx == end_idx:
            states_trace.append((token_idx, token_name, current_state, None))
            break

        if token_idx == start_idx:
            continue

        allowed      = allow_matrix[current_state, token_idx].item()
        state_before = current_state

        token_type = get_token_state(token_idx, action_space)
        found_type = STATE_NAMES.get(token_type, "END") if token_type is not None else "END"
        if not allowed:
            violations.append({
                "step":       step,
                "token_idx":  token_idx,
                "token_name": token_name,
                "state_name": STATE_NAMES[state_before],
                "found_type": found_type,
                "allowed":    allowed_types_str(state_before, action_space),
            })

        token_state = get_token_state(token_idx, action_space)
        if token_state == STATE_SO:
            next_state = STATE_SO_AFTER_POP if current_state == STATE_POP else STATE_SO
        elif token_state is not None:
            next_state = token_state
        else:
            next_state = current_state

        states_trace.append((token_idx, token_name, state_before, next_state))
        current_state = next_state

    return len(violations) == 0, violations, states_trace


# ── test runner ───────────────────────────────────────────────────────────────
def run_tests(sequences, action_space, verbose=False):
    n_valid, n_invalid = 0, 0

    for i, seq in enumerate(sequences):
        valid, violations, trace = check_sequence(seq, action_space)

        if valid:
            n_valid += 1
            if verbose:
                print(f"\n[{i:>4}] ✓  VALID   length={len(seq)}")
                for tok_idx, tok_name, s_before, s_after in trace:
                    after_str = STATE_NAMES.get(s_after, "END") if s_after is not None else "END"
                    print(f"         '{tok_name}' (idx={tok_idx:>3})  "
                          f"{STATE_NAMES.get(s_before,'?'):20s} → {after_str}")
        else:
            n_invalid += 1
            encoding = action_space.action_sequence_to_encoding(seq)
            print(f"\n[{i:>4}] ✗  INVALID  length={len(seq)}")
            print(f"       encoding : {encoding[:120]}")
            for v in violations:
                print(f"       step {v['step']:>3}: token '{v['token_name']}' "
                      f"(idx={v['token_idx']})  found [{v['found_type']}]  "
                      f"in state [{v['state_name']}]  "
                      f"— allowed here: [{v['allowed']}]")
            if verbose:
                print("       full trace:")
                for tok_idx, tok_name, s_before, s_after in trace:
                    after_str = STATE_NAMES.get(s_after, "END") if s_after is not None else "END"
                    print(f"         '{tok_name}' (idx={tok_idx:>3})  "
                          f"{STATE_NAMES.get(s_before,'?'):20s} → {after_str}")

    print("\n" + "=" * 60)
    print(f"Results: {n_valid} valid / {n_invalid} invalid / {len(sequences)} total")
    print("=" * 60)
    return n_valid, n_invalid


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Test GGS syntax masking using your actual project code")
    parser.add_argument("--project",
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help="Directory containing action_space.py "
                             "(default: same folder as this script)")
    parser.add_argument("--grammar",   required=True, help="Path to GGS grammar file")
    parser.add_argument("--sequences", required=True, help="Path to action sequences file")
    parser.add_argument("--verbose",   action="store_true",
                        help="Print full state trace for every sequence")
    args = parser.parse_args()

    # import YOUR actual code
    sys.path.insert(0, os.path.abspath(args.project))
    from action_space import Action_Space_GroupSelfies
    from group_selfies import GroupGrammar

    print(f"Project directory : {os.path.abspath(args.project)}")
    print(f"Loading grammar   : {args.grammar}")
    grammar      = GroupGrammar.from_file(args.grammar)
    action_space = Action_Space_GroupSelfies(grammar)
    print(f"  {action_space.N_actions} actions  |  "
          f"{action_space.n_groups} groups  |  "
          f"{action_space.n_si} coupling_in  |  "
          f"{action_space.n_so} shift_out  |  "
          f"idx_pop={action_space.idx_pop}  idx_end={action_space.idx_end}")

    # print allow matrix so you can visually verify it
    print("\nAllow matrix — permitted token types after each state:")
    for state_idx, state_name in STATE_NAMES.items():
        print(f"  [{state_idx}] {state_name:20s} → [{allowed_types_str(state_idx, action_space)}]")

    print(f"\nLoading sequences : {args.sequences}")
    sequences = load_sequences(args.sequences)
    print(f"  {len(sequences)} sequences loaded")

    if not sequences:
        print("No sequences to test. Exiting.")
        sys.exit(0)

    print("\n--- Per-sequence results ---")
    run_tests(sequences, action_space, verbose=args.verbose)


if __name__ == "__main__":
    main()