#!/usr/bin/env python3
"""Run the original NMO pretrainer with the three established compatibility fixes."""

import argparse
import configparser
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def prepare_source(path):
    source = path.read_text()
    fixes = (
        (
            'os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)',
            'os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))',
        ),
        ("    grammar = GroupGrammar(grammar_path)\n", ""),
        ("for epoch in range(1, epochs):", "for epoch in range(1, epochs + 1):"),
    )
    for old, new in fixes:
        if source.count(old) != 1:
            raise ValueError(f"NMO pretraining source changed near {old!r}; review before running")
        source = source.replace(old, new, 1)
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to pretraining configuration")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    config = configparser.ConfigParser()
    if not config.read(args.config):
        raise FileNotFoundError(args.config)
    output = Path(config.get("Pretrain", "output_path"))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new pretraining output_path: {output}")
    repo = ROOT / "NMO_source_code"
    framework = repo / "genetic_GFN_framework"
    sys.path[:0] = [
        str(framework),
        str(repo / "NMO"),
        str(repo),
        str(repo / "NMO/external_packages/group-selfies"),
    ]
    path = framework / "pretrain.py"
    # Execute a corrected in-memory copy; the original source file stays untouched.
    exec(compile(prepare_source(path), str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})


if __name__ == "__main__":
    main()
