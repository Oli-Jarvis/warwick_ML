"""
Create an empty REINVENT4 RNN prior for index-based (Group SELFIES) encoding.

Usage:
    python scripts/create_index_prior.py \
        --grammar_path my_grammar.json \
        --output index_prior_empty.prior \
        --num_layers 3 \
        --num_dimensions 512

The vocabulary size is derived automatically from the grammar file via
Action_Space_GroupSelfies.N_actions, so prior and scoring plugin always agree.
"""

import argparse
import sys
import torch

sys.path.insert(0, ".")   # make action_space.py importable from repo root

from reinvent.action_space import Action_Space_GroupSelfies
from reinvent.models.reinvent.models.vocabulary import IndexTokenizer, create_index_vocabulary
from reinvent.models.reinvent.models.model import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grammar_path", required=True,
                        help="Path to the GroupGrammar JSON file")
    parser.add_argument("--output", default="index_prior_empty.prior")
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_dimensions", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max_sequence_length", type=int, default=256)
    args = parser.parse_args()

    # Build action space from grammar — this defines N_actions and the vocabulary
    action_space = Action_Space_GroupSelfies.from_grammar_path(args.grammar_path)
    print(f"Action space size: {action_space.N_actions}")

    tokenizer = IndexTokenizer()
    vocabulary = create_index_vocabulary(action_space.N_actions, tokenizer)
    print(f"Vocabulary size: {len(vocabulary)}")

    network_params = {
    "num_layers": args.num_layers,
    "layer_size": args.num_dimensions,
    "dropout": args.dropout,
    "layer_normalization": False,
    }

    model = Model(
        vocabulary=vocabulary,
        tokenizer=tokenizer,
        meta_data=None,
        network_params=network_params,
        max_sequence_length=args.max_sequence_length,
    )

    torch.save({
        "model_type": model._model_type,
        "version": model._version,
        "vocabulary": vocabulary.get_dictionary(),
        "tokenizer": tokenizer,
        "network": model.network.state_dict(),
        "network_params": model.network.get_params(),
        "max_sequence_length": args.max_sequence_length,
    }, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
