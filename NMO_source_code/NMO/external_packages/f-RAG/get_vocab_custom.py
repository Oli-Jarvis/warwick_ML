# ---------------------------------------------------------------
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for f-RAG. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import re
import argparse
from collections import defaultdict
from tqdm import trange
import pandas as pd
import datamol as dm
import safe as sf
from rdkit import RDLogger
from fusion.slicer import MolSlicer
RDLogger.DisableLog('rdApp.*')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                        help='CSV file with a "smiles" column')
    parser.add_argument('--output', type=str, default='vocab/base.csv',
                        help='Output vocab CSV')
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    assert 'smiles' in df.columns, "CSV must have a 'smiles' column"
    print(f'Loaded {len(df)} molecules from {args.data}')

    slicer = MolSlicer(shortest_linker=True)

    frag2cnt = defaultdict(int)
    for i in trange(len(df), desc='Extracting fragments'):
        try:
            for safe_frag in slicer(df['smiles'].iloc[i]):
                if safe_frag is None:
                    continue
                smiles_frag = sf.decode(dm.to_smiles(safe_frag), remove_dummies=False)
                smiles_frag = re.sub(r'\[\d+\*\]', '[1*]', smiles_frag)
                if smiles_frag.count('*') not in {1, 2}:
                    continue
                frag2cnt[smiles_frag] += 1
        except KeyboardInterrupt:
            quit()
        except:
            continue

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    df_vocab = pd.DataFrame({'frag': list(frag2cnt.keys()),
                             'score': list(frag2cnt.values())})   # score = frequency
    df_vocab['size'] = df_vocab['frag'].apply(lambda frag: dm.to_mol(frag).GetNumAtoms())
    df_vocab = df_vocab.sort_values(by='score', ascending=False)
    df_vocab.to_csv(args.output, index=False)
    print(f'Wrote {len(df_vocab)} fragments to {args.output}')
    print(f'  arms    (1 attachment): {(df_vocab["frag"].str.count(r"[*]") == 1).sum()}')
    print(f'  linkers (2 attachments): {(df_vocab["frag"].str.count(r"[*]") == 2).sum()}')
