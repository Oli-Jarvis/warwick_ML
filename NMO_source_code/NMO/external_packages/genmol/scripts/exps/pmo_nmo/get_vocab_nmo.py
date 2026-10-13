import os
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
from genmol.utils.utils_chem import cut
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

if __name__ == '__main__':
    with open('./translated_smiles.smi') as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    print(f'Loaded {len(smiles_list)} SMILES')

    # collect all fragments
    frag2cnt = defaultdict(int)
    for smi in tqdm(smiles_list):
        frags = cut(smi)
        for frag in frags:
            frag2cnt[frag] += 1

    # uniform score
    df_vocab = pd.DataFrame({
        'frag':  list(frag2cnt.keys()),
        'score': 1.0,
    })
    df_vocab['size'] = df_vocab['frag'].apply(
        lambda f: Chem.MolFromSmiles(f).GetNumAtoms())

    foldername = 'scripts/exps/pmo/vocab'
    os.makedirs(foldername, exist_ok=True)
    df_vocab.to_csv(f'{foldername}/nmo_uniform.csv', index=False)
    print(f'Vocabulary size: {len(df_vocab)}')