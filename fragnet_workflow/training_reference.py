#!/usr/bin/env python3
"""Combined SMILES/GGS FragNet regression with audited stereo and holdouts.

Run from /storage/msszkb_grp/msshfg in the existing fragnet environment:
    python run_fragnet_combined.py
    python run_fragnet_combined.py --resume

Default inputs: every full_history.csv under directories whose names begin
with smiles or ggs, beneath --data-root (including archived copies). Exact
molecule/target observations are deduplicated before averaging distinct target
values. All input paths and rejected rows are exported. No gold stripping,
tautomer merging, salt removal, or invented stereoisomer enumeration occurs.

Ordinary tetrahedral and double-bond stereo are retained. Unknown stereo remains
unknown and is flagged; unsupported/enhanced stereo is rejected, never collapsed.
Native exp1s features are retained to keep pretrained dimensions compatible.
Runtime probes and graph checks fail if stereo features are missing. This is
not a mathematical guarantee that a GNN distinguishes every molecular graph.

The previous broad splits are REQUIRED: prior train/val molecular families
cannot become test data after hyperparameter selection. Stereo-free molecular
families are indivisible, with historical precedence train > val > test.
New families are assigned by a stable seeded hash to 80/10/10 splits. Fractions
are approximate because historical memberships and groups take precedence.
This is a molecule-family holdout, NOT a scaffold or chronological holdout.

--prepare-only builds audited graphs without training. --audit-only performs
cleaning/splitting without importing torch. --resume validates input/code hashes
before reusing graph shards and restores the last complete training epoch.
Graphs with failed 3D embedding or unsupported features are reported. Training
stops if any graph is rejected unless --allow-graph-rejections is explicit.

Final test evaluation happens only after validation-based early stopping and
restoration of the best checkpoint. No test metric is used for selection.
Requires the same packages as the supplied FragNet scripts, plus networkx
(already a dependency of PyTorch in normal installations).
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import math
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

HEAD = dict(h1=64, h2=1920, h3=1472, h4=2048, act='relu')
LR = 0.0007905475075930115
BROAD = 'fragnet_combined_log_P_upconversion_broad_optuna'
SPLITS = ('train', 'val', 'test')
REPO_REFERENCE = 'pnnl/FragNet bb9da67510f3ad725ff3e8b9789cb7567a0b5370'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', type=Path, default=Path('/storage/msszkb_grp/msshfg'))
    p.add_argument('--input-dir', type=Path, nargs='+', help='Explicit input roots/files, replacing automatic discovery.')
    p.add_argument('--broad-dir', type=Path)
    p.add_argument('--fragnet-root', type=Path)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--target-column', default='log_P_upconversion', help='Used as-is; no additional logarithm.')
    p.add_argument('--smiles-column', default='smiles')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--epochs', type=int, default=10000)
    p.add_argument('--patience', type=int, default=100)
    p.add_argument('--chunk-size', type=int, default=128)
    p.add_argument('--threads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK','1')))
    p.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--allow-graph-rejections', action='store_true')
    a=p.parse_args()
    a.data_root=a.data_root.expanduser().resolve()
    a.broad_dir=(a.broad_dir or a.data_root/BROAD).expanduser().resolve()
    a.fragnet_root=(a.fragnet_root or a.data_root/'smiles_baseline2'/'FragNet').expanduser().resolve()
    a.output_dir=(a.output_dir or a.data_root/'fragnet_combined_selected_stereo').expanduser().resolve()
    if min(a.epochs,a.patience,a.chunk_size,a.threads)<1 or not 0<=a.seed<2**31:
        p.error('Counts must be positive and seed must be in [0, 2**31).')
    return a


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def atomic_json(path, data):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False,default=str)+'\n')
    tmp.replace(path)


def atomic_pickle(path, data):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f: pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def molecule_identity(text):
    if not isinstance(text,str) or not text.strip(): raise ValueError('Missing/non-text SMILES')
    mol=Chem.MolFromSmiles(text.strip())
    if mol is None or mol.GetNumAtoms()==0: raise ValueError('Invalid/empty SMILES')
    # Atom map labels are bookkeeping, not molecular identity.
    for atom in mol.GetAtoms(): atom.SetAtomMapNum(0)
    Chem.AssignStereochemistry(mol,cleanIt=True,force=True)
    if mol.GetStereoGroups(): raise ValueError('Enhanced stereo groups require a richer representation')
    allowed={Chem.ChiralType.CHI_UNSPECIFIED,Chem.ChiralType.CHI_TETRAHEDRAL_CW,Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
    if any(x.GetChiralTag() not in allowed for x in mol.GetAtoms()):
        raise ValueError('Unsupported non-tetrahedral atom stereochemistry')
    stereo=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)
    # Remove only stereochemistry: retain isotopes, charges, and disconnected components.
    parent=Chem.Mol(mol); Chem.RemoveStereochemistry(parent)
    family=Chem.MolToSmiles(parent,canonical=True,isomericSmiles=True)
    unspecified=sum(str(info.specified)=='Unspecified' for info in Chem.FindPotentialStereo(mol))
    return stereo,family,int(unspecified)


def discover(a):
    files=set()
    roots=a.input_dir or [a.data_root]
    for root in roots:
        root=root.expanduser().resolve()
        if not root.exists(): raise FileNotFoundError(root)
        candidates=[root] if root.is_file() else root.rglob('full_history.csv')
        for path in candidates:
            path=path.resolve()
            if a.output_dir in path.parents or a.broad_dir in path.parents: continue
            if not a.input_dir:
                relative=path.relative_to(a.data_root)
                if not any(part.lower().startswith(('smiles','ggs')) for part in relative.parts[:-1]): continue
                if 'FragNet' in relative.parts: continue
            files.add(path)
    if not files: raise FileNotFoundError('No SMILES/GGS full_history.csv found; use --input-dir.')
    return sorted(files)


def read_table(path):
    # Sniff the header only: molecular SMILES commas must not influence delimiter choice.
    import csv
    with Path(path).open(encoding='utf-8-sig') as f: header=f.readline()
    sep=csv.Sniffer().sniff(header,delimiters=',;\t').delimiter
    frame=pd.read_csv(path,sep=sep,low_memory=False,encoding='utf-8-sig')
    frame.columns=frame.columns.str.strip()
    return frame


def clean_data(files,a):
    valid=[]; rejected=[]; counts=[]; cache={}
    for path in files:
        frame=read_table(path)
        required={a.smiles_column,a.target_column}
        if not required.issubset(frame.columns): raise ValueError(f'{path}: missing columns {required-set(frame.columns)}')
        before=len(valid)
        for rownum,(s,y) in enumerate(frame[[a.smiles_column,a.target_column]].itertuples(index=False,name=None),2):
            try:
                y=float(y)
                if not math.isfinite(y): raise ValueError('Missing/non-finite target')
                if not isinstance(s,str): raise ValueError('Missing/non-text SMILES')
                if s not in cache: cache[s]=molecule_identity(s)
                smi,fam,unknown=cache[s]
                valid.append(dict(smiles=smi,family=fam,y=y,unspecified_stereo=unknown,source_file=str(path),source_row=rownum))
            except (ValueError,TypeError,OverflowError) as e:
                rejected.append(dict(source_file=str(path),source_row=rownum,original_smiles=str(s),reason=str(e)))
        counts.append(dict(path=str(path),rows=len(frame),valid_rows=len(valid)-before))
        print(f'Loaded {path}: {len(valid)-before:,}/{len(frame):,} valid rows',flush=True)
    pd.DataFrame(rejected,columns=['source_file','source_row','original_smiles','reason']).to_csv(a.output_dir/'cleaning_rejected.csv',index=False)
    if not valid: raise ValueError('No valid rows remain.')
    observations=pd.DataFrame(valid)
    observations.to_csv(a.output_dir/'valid_observations.csv',index=False)
    # Repeated history entries and copied CSVs must not reweight the target mean.
    unique=observations.drop_duplicates(['smiles','y'])
    data=unique.groupby('smiles',as_index=False).agg(
        family=('family','first'),y=('y','mean'),target_sd=('y','std'),
        target_min=('y','min'),target_max=('y','max'),n_distinct_targets=('y','size'),
        unspecified_stereo=('unspecified_stereo','first'))
    data['target_sd']=data.target_sd.fillna(0)
    data=data.merge(observations.groupby('smiles',as_index=False).agg(
        observation_count=('y','size'),sources=('source_file',lambda s:json.dumps(sorted(set(s))))),on='smiles',validate='one_to_one')
    if len(data)<30: raise ValueError('Fewer than 30 unique molecules remain.')
    data.to_csv(a.output_dir/'cleaned_molecules.csv',index=False)
    data[data.n_distinct_targets>1].to_csv(a.output_dir/'duplicate_target_variation.csv',index=False)
    atomic_json(a.output_dir/'cleaning_summary.json',dict(files=counts,valid_observations=len(observations),unique_molecules=len(data),rejected_rows=len(rejected),unspecified_stereo_molecules=int((data.unspecified_stereo>0).sum()),aggregation='Mean of distinct target values per canonical isomeric SMILES'))
    return data


def historic_membership(broad_dir):
    memberships={}; exact={}; paths={}
    for name in SPLITS:
        path=broad_dir/'splits'/f'{name}.csv'
        if not path.exists(): raise FileNotFoundError(f'Required historical split: {path}')
        frame=pd.read_csv(path)
        if 'smiles' not in frame: raise ValueError(f'{path} lacks smiles')
        paths[name]=dict(path=str(path),sha256=digest(path),rows=len(frame))
        if frame.empty: raise ValueError(f'Historical {name} is empty')
        for text in frame.smiles:
            smi,fam,_=molecule_identity(text)
            memberships.setdefault(fam,set()).add(name)
            exact.setdefault(smi,set()).add(name)
    return memberships,exact,paths


def make_splits(data,memberships,seed):
    def assignment(family):
        seen=memberships.get(family,set())
        for name in SPLITS:  # train > validation > test
            if name in seen: return name
        number=int(hashlib.sha256(f'{seed}:{family}'.encode()).hexdigest()[:16],16)/2**64
        return 'train' if number<.8 else 'val' if number<.9 else 'test'
    data=data.copy(); data['split']=data.family.map(assignment)
    data['historical_splits']=data.family.map(lambda k:','.join(sorted(memberships.get(k,set()))))
    splits={name:data[data.split==name].sort_values('smiles').reset_index(drop=True) for name in SPLITS}
    verify_splits(splits,memberships)
    return splits


def verify_splits(splits,memberships):
    for name,frame in splits.items():
        if len(frame)<2: raise ValueError(f'{name} has fewer than two molecules; holdout is unusable.')
        if frame.smiles.duplicated().any(): raise ValueError(f'Duplicates within {name}')
        if not np.isfinite(frame.y.to_numpy(float)).all(): raise ValueError('Non-finite labels')
    for i,left in enumerate(SPLITS):
        for right in SPLITS[i+1:]:
            for key in ('smiles','family'):
                if set(splits[left][key]) & set(splits[right][key]): raise ValueError(f'{key} leakage: {left}/{right}')
    if any(memberships.get(fam,set()) & {'train','val'} for fam in splits['test'].family):
        raise ValueError('Historical training/validation data entered test!')
    if any('train' in memberships.get(fam,set()) for fam in splits['val'].family):
        raise ValueError('Historical training data entered validation!')


def make_config(a):
    path=a.broad_dir/'fragnet_broad_optimised.yaml'
    config=yaml.safe_load(path.read_text())
    model=config['finetune']['model']
    expected={'num_layer':4,'num_heads':4,'emb_dim':128,'fthead':'FTHead3'}
    for key,value in expected.items():
        if model.get(key)!=value: raise ValueError(f'Broad config {key}={model.get(key)!r}, expected {value!r}')
    if tuple(config.get(k) for k in ('atom_features','frag_features','edge_features'))!=(167,167,17):
        raise ValueError('Expected native exp1s features 167/167/17.')
    if config.get('model_version')!='gat2': raise ValueError('Expected gat2 backbone')
    model.update(HEAD,drop_ratio=.25)
    config['finetune'].update(lr=LR,batch_size=128,target_type='regr',loss='mse',use_schedular=False,n_epochs=a.epochs,es_patience=a.patience)
    config['seed']=a.seed; config['exp_dir']=str(a.output_dir/'experiment')
    config['finetune']['chkpoint_name']=str(a.output_dir/'experiment'/'ft.pt')
    for name in SPLITS: config['finetune'][name]={'path':str(a.output_dir/'graph_data'/f'{name}.pkl')}
    cp=config.get('pretrain',{}).get('chkpoint_name')
    if cp:
        cp=Path(cp).expanduser()
        if not cp.is_absolute():
            choices=[a.broad_dir/cp,a.data_root/cp,a.fragnet_root/cp,a.fragnet_root/'fragnet'/cp]
            cp=next((v for v in choices if v.exists()),cp)
        if not cp.exists(): raise FileNotFoundError(f'Broad pretraining checkpoint unavailable: {cp}')
        config['pretrain']['chkpoint_name']=str(cp.resolve())
    # Native checkpoint loading also validates dimensions strictly.
    return config


def native_feature_audit():
    from fragnet.dataset.features import FeaturesEXP
    f=FeaturesEXP()
    if not f.use_bond_chirality: raise RuntimeError('Bond stereochemistry is disabled')
    def atom(s):
        m=Chem.MolFromSmiles(s)
        a=next(a for a in m.GetAtoms() if a.GetChiralTag()!=Chem.ChiralType.CHI_UNSPECIFIED)
        return f.atom_features_one_hot(a)
    def bond(s):
        m=Chem.MolFromSmiles(s)
        b=next(b for b in m.GetBonds() if b.GetBondType()==Chem.BondType.DOUBLE)
        return f.bond_features_one_hot(b,use_chirality=True)
    r,s=atom('N[C@H](C)C(=O)O'),atom('N[C@@H](C)C(=O)O')
    e,z=bond('F/C=C/F'),bond('F/C=C\\F')
    if len(r)!=167 or len(e)!=17: raise RuntimeError('Native feature dimensions changed')
    if np.array_equal(r,s) or np.array_equal(e,z): raise RuntimeError('Native atom/bond stereo features collapse stereoisomers')
    return dict(atom_stereo_probe=True,bond_stereo_probe=True,atom_features=167,bond_features=17)


def make_3d(smiles,seed):
    m=Chem.AddHs(Chem.MolFromSmiles(smiles))
    for random_coords in (False,True):
        settings=AllChem.ETKDGv3(); settings.randomSeed=seed
        settings.enforceChirality=True; settings.useRandomCoords=random_coords; settings.numThreads=1
        if AllChem.EmbedMolecule(m,settings)==0: break
    else: raise ValueError('3D embedding failed; no silent 2D fallback')
    if AllChem.MMFFHasAllMoleculeParams(m): AllChem.MMFFOptimizeMolecule(m,maxIters=200)
    # Retain labels, and independently verify specified stereo against coordinates.
    encoded=molecule_identity(Chem.MolToSmiles(Chem.RemoveHs(m),isomericSmiles=True))[0]
    if encoded!=smiles: raise ValueError('Stereo/identity changed while adding coordinates')
    check=Chem.Mol(m); Chem.AssignStereochemistryFrom3D(check,confId=0,replaceExistingTags=True)
    Chem.AssignStereochemistry(m,cleanIt=True,force=True)
    Chem.AssignStereochemistry(check,cleanIt=True,force=True)
    for a,b in zip(m.GetAtoms(),check.GetAtoms()):
        if a.HasProp('_CIPCode') and (not b.HasProp('_CIPCode') or a.GetProp('_CIPCode')!=b.GetProp('_CIPCode')):
            raise ValueError('3D embedding inverted a specified stereocentre')
    for a,b in zip(m.GetBonds(),check.GetBonds()):
        if a.GetStereo() in (Chem.BondStereo.STEREOE,Chem.BondStereo.STEREOZ) and a.GetStereo()!=b.GetStereo():
            raise ValueError('3D embedding changed specified double-bond stereo')
    return m


def validate_graph(g,smiles,y):
    import torch
    if g is None: raise ValueError('FragNet returned no graph')
    if molecule_identity(g.smiles)[0]!=smiles: raise ValueError('Graph identity mismatch')
    if g.x_atoms.ndim!=2 or g.x_atoms.shape[1]!=167 or g.x_frags.shape[1]!=167 or g.edge_attr.shape[1]!=17:
        raise ValueError('Graph feature dimensions mismatch')
    if g.y.numel()!=1 or not np.isclose(float(g.y.item()),y,rtol=1e-6,atol=1e-6): raise ValueError('Graph label mismatch')
    for key,value in g:
        if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f'Non-finite graph tensor: {key}')


def one_graph(row,creator,seed):
    m=make_3d(row.smiles,seed)
    graph=creator.create_data_point([row.smiles,[float(row.y)],m,m.GetConformer(), 'brics'])
    validate_graph(graph,row.smiles,row.y)
    # FragmentedMol may add wedges. Compare against the SAME post-fragmentation
    # molecule, not an independently reordered SMILES parse.
    f=creator.feature_creator
    atoms,edges,bonds=f.get_atom_and_bond_features_atom_graph_one_hot(m,True)
    if not np.array_equal(graph.x_atoms.cpu().numpy(),np.asarray(atoms,dtype=np.float32)):
        raise ValueError('Graph lost or reordered atom features')
    if not np.array_equal(graph.edge_index.cpu().numpy(),np.asarray(edges)) or not np.array_equal(graph.edge_attr.cpu().numpy(),np.asarray(bonds,dtype=np.float32)):
        raise ValueError('Graph lost or reordered bond features')
    return graph


def feature_graph(g):
    import networkx as nx
    graph=nx.DiGraph()
    for i,values in enumerate(g.x_atoms.cpu().numpy()): graph.add_node(i,label=values.tobytes().hex())
    for pair,values in zip(g.edge_index.t().cpu().tolist(),g.edge_attr.cpu().numpy()): graph.add_edge(*pair,label=values.tobytes().hex())
    return graph


def audit_stereo_siblings(graphs,frame):
    # Exact labelled graph-isomorphism comparison for siblings: atom ordering
    # and arbitrary conformer differences must not be the only distinction.
    import networkx as nx
    lookup={g.smiles:g for g in graphs}; collisions=[]
    for family,group in frame.groupby('family'):
        if len(group)<2: continue
        encoded=[]
        for smiles in group.smiles:
            fg=feature_graph(lookup[smiles])
            for other,previous in encoded:
                if nx.is_isomorphic(fg,previous,node_match=lambda a,b:a['label']==b['label'],edge_match=lambda a,b:a['label']==b['label']):
                    collisions.append(dict(smiles_a=other,smiles_b=smiles,family=family))
            encoded.append((smiles,fg))
    return collisions


def prepare_graphs(splits,a,memberships):
    from fragnet.dataset.data import CreateData
    creator=CreateData(data_type='exp1s',create_bond_graph_data=True,add_dhangles=False)
    probes=[('N[C@H](C)C(=O)O','N[C@@H](C)C(=O)O'),('F/C=C/F','F/C=C\\F')]
    for pair in probes:
        rows=pd.DataFrame([dict(smiles=molecule_identity(s)[0],y=0.,family=molecule_identity(s)[1]) for s in pair])
        with open(os.devnull,'w') as quiet,contextlib.redirect_stdout(quiet):
            graphs=[one_graph(r,creator,a.seed) for r in rows.itertuples()]
        if audit_stereo_siblings(graphs,rows): raise RuntimeError('End-to-end graph stereo probe failed')
    rejected=[]; accepted={}; all_collisions=[]
    for name,frame in splits.items():
        graphs=[]
        for start in range(0,len(frame),a.chunk_size):
            chunk=frame.iloc[start:start+a.chunk_size]
            shard=a.output_dir/'graph_data'/f'{name}_{start:08d}.pkl'
            if shard.exists():
                with shard.open('rb') as f: payload=pickle.load(f)
            else:
                payload={'graphs':[],'rejected':[]}
                for row in chunk.itertuples():
                    try:
                        with open(os.devnull,'w') as quiet,contextlib.redirect_stdout(quiet):
                            g=one_graph(row,creator,a.seed)
                        payload['graphs'].append(g)
                    except Exception as e:
                        payload['rejected'].append(dict(split=name,smiles=row.smiles,y=float(row.y),reason=f'{type(e).__name__}: {e}'))
                atomic_pickle(shard,payload)
            wanted=dict(zip(chunk.smiles,chunk.y)); got=[]
            for g in payload['graphs']:
                if g.smiles not in wanted: raise ValueError('Stale graph shard identity')
                validate_graph(g,g.smiles,wanted[g.smiles]); got.append(g.smiles)
            covered=got+[r['smiles'] for r in payload['rejected']]
            if len(covered)!=len(set(covered)) or set(covered)!=set(wanted): raise ValueError('Graph shard coverage mismatch')
            graphs.extend(payload['graphs']); rejected.extend(payload['rejected'])
            print(f'{name} graphs: {min(start+a.chunk_size,len(frame)):,}/{len(frame):,}; accepted {len(graphs):,}',flush=True)
        found={g.smiles for g in graphs}
        accepted[name]=frame[frame.smiles.isin(found)].reset_index(drop=True)
        all_collisions.extend(audit_stereo_siblings(graphs,accepted[name]))
        atomic_pickle(a.output_dir/'graph_data'/f'{name}.pkl',graphs)
        accepted[name].to_csv(a.output_dir/'splits'/f'{name}.csv',index=False)
    pd.DataFrame(rejected,columns=['split','smiles','y','reason']).to_csv(a.output_dir/'graph_rejected.csv',index=False)
    pd.DataFrame(all_collisions,columns=['smiles_a','smiles_b','family']).to_csv(a.output_dir/'stereo_feature_collisions.csv',index=False)
    if all_collisions: raise RuntimeError('Distinct stereo identities share identical labelled atom/bond graphs; see stereo_feature_collisions.csv. Training blocked.')
    verify_splits(accepted,memberships)
    atomic_json(a.output_dir/'split_audit.json',dict(accepted_counts={k:len(v) for k,v in accepted.items()},accepted_fractions={k:len(v)/sum(map(len,accepted.values())) for k,v in accepted.items()},exact_overlap=0,stereo_family_overlap=0,historical_train_val_in_test=0,graph_rejections=len(rejected),stereo_feature_collisions=0))
    if rejected and not a.allow_graph_rejections:
        raise RuntimeError(f'{len(rejected)} graph(s) rejected. Inspect graph_rejected.csv; to accept these exclusions rerun with --resume --allow-graph-rejections.')
    return accepted
def seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def torch_load(path: Path, device: Any, torch: Any) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)

def unwrap_state_dict(value: Any) -> Any:
    if isinstance(value, dict):
        for key in ("state_dict", "model_state_dict"):
            nested = value.get(key)
            if isinstance(nested, dict):
                return nested
    return value

def build_model(
    config: dict[str, Any],
    head_params: dict[str, Any],
) -> Any:
    from fragnet.model.gat.gat2 import FragNetFineTune

    model_config = config["finetune"]["model"]
    return FragNetFineTune(
        n_classes=int(model_config["n_classes"]),
        atom_features=int(config["atom_features"]),
        frag_features=int(config["frag_features"]),
        edge_features=int(config["edge_features"]),
        num_heads=int(model_config["num_heads"]),
        num_layer=int(model_config["num_layer"]),
        drop_ratio=float(model_config["drop_ratio"]),
        emb_dim=int(model_config.get("emb_dim", 128)),
        h1=int(head_params["h1"]),
        h2=int(head_params["h2"]),
        h3=int(head_params["h3"]),
        h4=int(head_params["h4"]),
        act=str(head_params["act"]),
        fthead="FTHead3",
    )

def load_pretrained_weights(
    model: Any,
    config: dict[str, Any],
    device: Any,
    torch: Any,
) -> None:
    checkpoint_text = config.get("pretrain", {}).get("chkpoint_name")
    if not checkpoint_text:
        return
    checkpoint = Path(str(checkpoint_text)).expanduser()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint}")

    from fragnet.model.gat.gat2_pretrain import FragNetPreTrain

    pretrain = config["pretrain"]
    pretrained_model = FragNetPreTrain(
        atom_features=int(config["atom_features"]),
        frag_features=int(config["frag_features"]),
        edge_features=int(config["edge_features"]),
        num_layer=int(pretrain["num_layer"]),
        drop_ratio=float(pretrain["drop_ratio"]),
        num_heads=int(pretrain["num_heads"]),
        emb_dim=int(pretrain["emb_dim"]),
    )
    state = unwrap_state_dict(torch_load(checkpoint, device, torch))
    pretrained_model.load_state_dict(state)
    model.pretrain.load_state_dict(pretrained_model.pretrain.state_dict())
    del pretrained_model, state

def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None

def regression_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    true = np.asarray(true, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    if len(true) != len(pred):
        raise ValueError("Prediction and target lengths differ.")
    if len(true) == 0:
        raise ValueError("Cannot score an empty dataset.")
    residual = pred - true
    mse = float(np.mean(residual**2))
    mae = float(np.mean(np.abs(residual)))
    denominator = float(np.sum((true - np.mean(true)) ** 2))
    r2 = (
        float(1.0 - np.sum(residual**2) / denominator)
        if denominator > 0
        else math.nan
    )
    pearson = (
        float(np.corrcoef(true, pred)[0, 1])
        if len(true) > 1 and np.std(true) > 0 and np.std(pred) > 0
        else math.nan
    )
    true_rank = pd.Series(true).rank(method="average").to_numpy()
    pred_rank = pd.Series(pred).rank(method="average").to_numpy()
    spearman = (
        float(np.corrcoef(true_rank, pred_rank)[0, 1])
        if len(true) > 1 and np.std(true_rank) > 0 and np.std(pred_rank) > 0
        else math.nan
    )
    return {
        "n": int(len(true)),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": mae,
        "r2": finite_or_none(r2),
        "pearson_r": finite_or_none(pearson),
        "spearman_rho": finite_or_none(spearman),
    }

def load_graphs(path,frame):
    with path.open('rb') as f: graphs=pickle.load(f)
    if len(graphs)!=len(frame) or len({g.smiles for g in graphs})!=len(graphs): raise ValueError('Graph count/uniqueness mismatch')
    lookup=frame.set_index('smiles')
    if set(lookup.index)!={g.smiles for g in graphs}: raise ValueError('Split/graph identity mismatch')
    for g in graphs: validate_graph(g,g.smiles,float(lookup.loc[g.smiles,'y']))
    return graphs


def save_torch(path,data,torch):
    tmp=path.with_suffix(path.suffix+'.tmp'); torch.save(data,tmp); tmp.replace(path)


def predict(model,loader,device,torch):
    model.eval(); true=[]; predictions=[]
    with torch.no_grad():
        for batch in loader:
            batch={k:v.to(device) for k,v in batch.items()}
            out=model(batch).reshape(-1); y=batch['y'].reshape(-1)
            if out.shape!=y.shape or not torch.isfinite(out).all(): raise RuntimeError('Invalid model predictions')
            true.extend(y.cpu().numpy().tolist()); predictions.extend(out.cpu().numpy().tolist())
    true=np.array(true); predictions=np.array(predictions)
    return regression_metrics(true,predictions),true,predictions


def train(a,config,splits,fingerprint):
    import torch
    from torch.utils.data import DataLoader
    from fragnet.dataset.data import collate_fn
    torch.set_num_threads(a.threads); seed_everything(a.seed,torch)
    if a.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('Requested CUDA is unavailable')
    device=torch.device('cuda' if a.device=='auto' and torch.cuda.is_available() else ('cpu' if a.device=='auto' else a.device))
    print(f'Training device: {device}; threads: {a.threads}',flush=True)
    config['device']=str(device)
    (a.output_dir/'fragnet_selected.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    model=build_model(config,HEAD)
    experiment=a.output_dir/'experiment'; best_path=experiment/'ft.pt'; last_path=experiment/'last.pt'
    if not last_path.exists(): load_pretrained_weights(model,config,device,torch)
    model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)
    train_data=load_graphs(a.output_dir/'graph_data'/'train.pkl',splits['train'])
    val_data=load_graphs(a.output_dir/'graph_data'/'val.pkl',splits['val'])
    # All training molecules are used every epoch, including the last partial batch.
    loader=DataLoader(train_data,batch_size=128,shuffle=True,drop_last=False,collate_fn=collate_fn,num_workers=0)
    val_loader=DataLoader(val_data,batch_size=128,shuffle=False,collate_fn=collate_fn,num_workers=0)
    best=math.inf; bad=0; start=0; best_epoch=None; history=[]; best_state=None
    if last_path.exists():
        # This file is created by this script in the explicitly resumed output directory.
        saved=torch.load(last_path,map_location=device,weights_only=False)
        if saved['fingerprint']!=fingerprint: raise RuntimeError('Training checkpoint does not match inputs/settings')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        best_state={k:v.detach().cpu().clone() for k,v in saved['best_model'].items()}
        save_torch(best_path,best_state,torch)
        best=saved['best']; best_epoch=saved['best_epoch']; bad=saved['bad']; start=saved['epoch']+1; history=saved['history']
        random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng']); torch.set_rng_state(saved['torch_rng'].cpu())
        if device.type=='cuda' and saved['cuda_rng'] is not None: torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
        print(f'Resuming after epoch {start}; best validation MSE {best:.6f}',flush=True)
    for epoch in range(start,a.epochs):
        if bad>=a.patience: break
        model.train(); squared_error=0.; count=0
        for batch in loader:
            batch={k:v.to(device) for k,v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            out=model(batch).reshape(-1); y=batch['y'].reshape(-1)
            if out.shape!=y.shape: raise RuntimeError('Prediction/target shape mismatch')
            loss=torch.nn.functional.mse_loss(out,y)
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite training loss')
            loss.backward(); optimizer.step()
            squared_error+=float(loss.detach())*y.numel(); count+=y.numel()
        metrics,_,_=predict(model,val_loader,device,torch)
        score=metrics['mse']
        if score<best:
            best=score; bad=0; best_epoch=epoch+1
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            save_torch(best_path,best_state,torch)
        else: bad+=1
        history.append(dict(epoch=epoch+1,training_mse=squared_error/count,validation_mse=score,best_validation_mse=best))
        # Complete epoch checkpoint is the resume authority.
        save_torch(last_path,dict(fingerprint=fingerprint,epoch=epoch,model=model.state_dict(),best_model=best_state,optimizer=optimizer.state_dict(),best=best,best_epoch=best_epoch,bad=bad,history=history,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None),torch)
        pd.DataFrame(history).to_csv(experiment/'training_history.csv',index=False)
        print(f'Epoch {epoch+1}: train MSE={squared_error/count:.6f}; val MSE={score:.6f}; best={best:.6f}; patience={bad}/{a.patience}',flush=True)
    if not best_path.exists(): raise RuntimeError('No best checkpoint exists')
    model.load_state_dict(torch_load(best_path,device,torch))
    # Test is loaded only now, once model/epoch selection is complete.
    results={}
    for name in SPLITS:
        data=train_data if name=='train' else val_data if name=='val' else load_graphs(a.output_dir/'graph_data'/'test.pkl',splits['test'])
        evaluation=DataLoader(data,batch_size=128,shuffle=False,collate_fn=collate_fn,num_workers=0)
        metrics,true,pred=predict(model,evaluation,device,torch)
        results[name]=metrics
        pd.DataFrame(dict(smiles=[g.smiles for g in data],true=true,predicted=pred,residual=pred-true)).to_csv(experiment/f'{name}_predictions.csv',index=False)
    atomic_json(experiment/'regression_metrics.json',dict(best_epoch=best_epoch,metrics=results,test_used_for_selection=False,fingerprint=fingerprint,checkpoint_sha256=digest(best_path)))
    print(json.dumps(results,indent=2),flush=True)


def main():
    a=parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()) and not a.resume:
        raise FileExistsError('Output directory is nonempty. Use --resume for this same run or choose a new --output-dir.')
    for sub in ('','splits','graph_data','experiment'): (a.output_dir/sub).mkdir(parents=True,exist_ok=True)
    files=discover(a)
    membership,exact,historical_paths=historic_membership(a.broad_dir)
    config=make_config(a)
    # Hash data, historical splits, model config, script, and native feature/model
    # implementation. Cached graphs cannot silently outlive their inputs.
    native_sources=sorted((a.fragnet_root/'fragnet'/'dataset').glob('*.py'))+sorted((a.fragnet_root/'fragnet'/'model'/'gat').glob('*.py'))
    if not native_sources: raise FileNotFoundError(f'FragNet repository unavailable: {a.fragnet_root}')
    stable_config=json.loads(json.dumps(config))
    stable_config['finetune'].pop('n_epochs',None)  # may extend unfinished run on resume
    stable_config['finetune'].pop('es_patience',None)
    signature=dict(inputs={str(p):digest(p) for p in files},historical_splits=historical_paths,config=stable_config,seed=a.seed,chunk_size=a.chunk_size,smiles_column=a.smiles_column,target_column=a.target_column,script_sha256=digest(Path(__file__)),native_sources={str(p):digest(p) for p in native_sources},rdkit_version=rdBase.rdkitVersion)
    cp=config.get('pretrain',{}).get('chkpoint_name')
    signature['pretraining_sha256']=digest(cp) if cp else None
    fingerprint=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    manifest_path=a.output_dir/'run_manifest.json'
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old['fingerprint']!=fingerprint: raise RuntimeError('Inputs, split history, code, or configuration changed. Use a new output directory.')
    atomic_json(manifest_path,dict(fingerprint=fingerprint,signature=signature,selected_head=HEAD,learning_rate=LR,dropout=.25,batch_size=128,split_policy='historical precedence train > val > test, grouped by stereo-free molecule',stereo_scope='specified tetrahedral and double-bond; unspecified remains unspecified',coordinate_policy='ETKDGv3 enforced chirality; MMFF when available; no 2D fallback',target_policy='mean of distinct observed target values per canonical isomeric SMILES',reference=REPO_REFERENCE))
    data=clean_data(files,a)
    splits=make_splits(data,membership,a.seed)
    for name,frame in splits.items():
        frame.to_csv(a.output_dir/'splits'/f'{name}_requested.csv',index=False)
        print(f'{name}: {len(frame):,} molecules in {frame.family.nunique():,} stereo-free families',flush=True)
    conflicts=[dict(family=k,prior_splits=','.join(sorted(v)),assigned=next(n for n in SPLITS if n in v)) for k,v in membership.items() if len(v)>1]
    pd.DataFrame(conflicts,columns=['family','prior_splits','assigned']).to_csv(a.output_dir/'historical_group_conflicts.csv',index=False)
    (a.output_dir/'fragnet_selected.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    if a.audit_only:
        print('Identity and split audit complete. No graphs or training performed.'); return
    sys.path.insert(0,str(a.fragnet_root))
    import torch
    torch.set_num_threads(a.threads)
    atomic_json(a.output_dir/'stereo_feature_audit.json',native_feature_audit())
    splits=prepare_graphs(splits,a,membership)
    if a.prepare_only:
        print('Graph preparation complete. Train with the same arguments plus --resume, without --prepare-only.'); return
    if (a.output_dir/'experiment'/'regression_metrics.json').exists():
        print('Final evaluation already exists. This completed run will not retrain or repeatedly evaluate test.'); return
    train(a,config,splits,fingerprint)


