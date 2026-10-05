"""Shared FragNet molecule, graph and model construction functions."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def molecule_identity(text):
    if not isinstance(text,str) or not text.strip(): raise ValueError('Missing/non-text SMILES')
    mol=Chem.MolFromSmiles(text.strip())
    if mol is None or mol.GetNumAtoms()==0: raise ValueError('Invalid/empty SMILES')

    for atom in mol.GetAtoms(): atom.SetAtomMapNum(0)
    Chem.AssignStereochemistry(mol,cleanIt=True,force=True)
    if mol.GetStereoGroups(): raise ValueError('Enhanced stereo groups require a richer representation')
    allowed={Chem.ChiralType.CHI_UNSPECIFIED,Chem.ChiralType.CHI_TETRAHEDRAL_CW,Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
    if any(x.GetChiralTag() not in allowed for x in mol.GetAtoms()):
        raise ValueError('Unsupported non-tetrahedral atom stereochemistry')
    stereo=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)

    parent=Chem.Mol(mol); Chem.RemoveStereochemistry(parent)
    family=Chem.MolToSmiles(parent,canonical=True,isomericSmiles=True)
    unspecified=sum(str(info.specified)=='Unspecified' for info in Chem.FindPotentialStereo(mol))
    return stereo,family,int(unspecified)

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


    f=creator.feature_creator
    atoms,edges,bonds=f.get_atom_and_bond_features_atom_graph_one_hot(m,True)
    if not np.array_equal(graph.x_atoms.cpu().numpy(),np.asarray(atoms,dtype=np.float32)):
        raise ValueError('Graph lost or reordered atom features')
    if not np.array_equal(graph.edge_index.cpu().numpy(),np.asarray(edges)) or not np.array_equal(graph.edge_attr.cpu().numpy(),np.asarray(bonds,dtype=np.float32)):
        raise ValueError('Graph lost or reordered bond features')
    return graph

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
