"""
Extract index sequences from a MolTensorDataset (.pt file created by dataset.py)
into a flat text file for REINVENT4 Transfer Learning.

The dataset stores padded action sequences — this script strips the padding
and writes one space-separated index sequence per line.

Usage:
    python scripts/extract_dataset_for_tl.py \
        --dataset my_dataset.pt \
        --output my_molecules_indices.smi \
        --grammar_path my_grammar.json
"""

import argparse
import torch
import sys
sys.path.insert(0, ".")

from reinvent.action_space import Action_Space_GroupSelfies


# Must match the class as saved by dataset.py and be in __main__
class MolTensorDataset(torch.utils.data.TensorDataset):
    def __init__(self, *tensors, meta_data={}):
        super().__init__(*tensors)
        self.meta_data_dict = meta_data

sys.modules[__name__].MolTensorDataset = MolTensorDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      required=True, help="Path to .pt dataset file")
    parser.add_argument("--output",       required=True, help="Output text file for REINVENT TL")
    parser.add_argument("--grammar_path", required=True)
    args = parser.parse_args()

    # Load action space to get end token index
    action_space = Action_Space_GroupSelfies.from_grammar_path(args.grammar_path)
    end_token_idx = action_space.reversed_action_space["End"]

    # Load dataset — padded_sequences is always the first tensor
    dataset = torch.load(args.dataset, weights_only=False)
    padded_sequences = dataset.tensors[0]   # shape: (N, max_seq_length)

    print(f"Dataset size:      {len(padded_sequences)} sequences")
    print(f"Max sequence length: {padded_sequences.shape[1]}")
    print(f"End token index:   {end_token_idx}")

    n_written = 0
    with open(args.output, "w") as f:
        for seq in padded_sequences:
            # Strip padding: keep tokens up to and including the first End token
            tokens = seq.tolist()
            try:
                end_pos = tokens.index(end_token_idx)
                tokens = tokens[:end_pos + 1]   # include End token
            except ValueError:
                pass   # no End token found — use full sequence

            index_str = " ".join(str(int(t)) for t in tokens)
            f.write(index_str + "\n")
            n_written += 1

    print(f"Written {n_written} sequences -> {args.output}")


if __name__ == "__main__":
    main()