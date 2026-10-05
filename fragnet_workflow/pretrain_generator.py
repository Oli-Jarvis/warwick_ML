#!/usr/bin/env python3
"""Step 1: pretrain or reuse the RNN generator prior."""
import json
import sys
import time

from generator import call, digest, environment, paths_for, prepare, runtime_check
from settings import parse_args


def prior_signature(args, paths):
    return dict(dataset_sha256=digest(paths['dataset']), vocabulary_sha256=digest(paths['vocabulary']),
                pretrain_source_sha256=digest(paths['code']/'pretrain.py'), epochs=args.epochs,
                model='rnn', num_layers=3, d_model=512, max_seq_length=140, seed=args.seed)


def pretrain(args, paths):
    signature = prior_signature(args, paths)
    record = paths['prior_dir']/'prior_manifest.json'
    if args.prior:
        if not paths['prior'].is_file():
            raise FileNotFoundError(paths['prior'])
        print(f'Using supplied prior: {paths["prior"]}', flush=True)
        return
    if paths['prior'].is_file():
        if not record.is_file():
            raise ValueError('Prior exists without completion record. Check pretrain.log before reusing it; use --prior explicitly if validated.')
        manifest = json.loads(record.read_text())
        if manifest['signature'] != signature or manifest['checkpoint_sha256'] != digest(paths['prior']):
            raise ValueError('Prior settings/hash differ from this request. Use a new --work directory or explicitly supply a compatible --prior.')
        print(f'Reusing completed generator prior: {paths["prior"]}', flush=True)
        return
    paths['prior_dir'].mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    call([sys.executable, paths['code']/'pretrain.py', paths['config'], '--seed', args.seed],
         paths['work']/'pretrain.log', paths['framework'], environment(args))
    if not paths['prior'].is_file():
        raise RuntimeError('Pretraining ended without prior.pt')
    record.write_text(json.dumps(dict(signature=signature, checkpoint_sha256=digest(paths['prior']),
                                    elapsed_seconds=time.monotonic()-start), indent=2)+'\n')


def main(argv=None):
    args = parse_args('pretrain', argv)
    paths = paths_for(args)
    prepare(args, paths)
    runtime_check(args, paths)
    pretrain(args, paths)


if __name__ == '__main__':
    main()
