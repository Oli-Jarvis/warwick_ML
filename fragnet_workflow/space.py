"""Fit one reference map; transform and assign candidates without refitting."""
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch, AgglomerativeClustering
from sklearn.metrics import pairwise_distances_argmin_min


def fit_coordinates(bonding, structural, prop, components=3, weight=1., threshold=.5, clusters=20):
    if components < 1 or components > min(len(bonding)-1, bonding.shape[1], structural.shape[1]):
        raise ValueError('Requested PCA component count is incompatible with reference descriptors')
    if weight <= 0 or threshold <= 0 or clusters < 2:
        raise ValueError('Weight/threshold must be positive; at least two clusters required')
    bundle = dict(components=components, weight=weight)
    coordinates = []
    for label, matrix in [('bonding', bonding), ('structural', structural)]:
        scaler = StandardScaler().fit(matrix)
        pca = PCA(n_components=components, svd_solver='randomized', random_state=42)
        pca.fit(scaler.transform(matrix))
        pc = pca.transform(scaler.transform(matrix))
        bundle[label] = dict(scaler=scaler, pca=pca)
        coordinates.append(pc)
    coords = np.column_stack([*coordinates, prop])
    mean = coords.mean(axis=0)
    scales = np.r_[np.repeat(coords[:,0].std(), components),
                   np.repeat(coords[:,components].std(), components), coords[:,-1].std()/weight]
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError('Reference PC1/property must have nonzero finite variance')
    matrix = (coords-mean)/scales
    birch = Birch(threshold=threshold, n_clusters=None).fit(matrix)
    centres = birch.subcluster_centers_
    if len(centres) < clusters:
        raise ValueError(f'Only {len(centres)} BIRCH subclusters; lower --threshold or --clusters')
    labels = AgglomerativeClustering(n_clusters=clusters, linkage='ward').fit_predict(centres)
    sublabels, _ = pairwise_distances_argmin_min(matrix, centres)
    bundle.update(mean=mean, scales=scales, centres=centres, subcluster_labels=labels,
                  reference_matrix=matrix)
    return bundle, coords, labels[sublabels]


def transform_coordinates(bundle, bonding, structural, prop):
    pcs = []
    for name, values in [('bonding', bonding), ('structural', structural)]:
        reference_fit = bundle[name]
        projected = reference_fit['pca'].transform(reference_fit['scaler'].transform(values))[:,:bundle['components']]
        if 'coordinate_scale' in reference_fit:
            projected = projected * reference_fit['coordinate_scale'] + reference_fit['coordinate_offset']
        pcs.append(projected)
    coords = np.column_stack([*pcs, prop])
    matrix = (coords-bundle['mean'])/bundle['scales']
    sub, distance = pairwise_distances_argmin_min(matrix, bundle['centres'])
    nearest, ref_distance = pairwise_distances_argmin_min(matrix, bundle['reference_matrix'])
    labels = (bundle['reference'].cluster.to_numpy()[nearest]
              if bundle.get('assignment') == 'nearest_reference' else bundle['subcluster_labels'][sub])
    return coords, labels, distance, nearest, ref_distance


def describe(frame, out, bundle=None):
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.ML.Descriptors import MoleculeDescriptors
    from dscribe.descriptors import SOAP
    import conformers as geometry
    frame = frame.copy().reset_index(drop=True)
    frame['chemplot_smiles'] = frame.smiles.str.strip().str.replace('[Au]', '', regex=False)
    # Preserve input alignment when prediction or geometry fails.
    frame['_row'] = np.arange(len(frame))
    valid = frame.chemplot_smiles.map(lambda s: bool(s) and Chem.MolFromSmiles(s) is not None)
    errors = [dict(row=int(i), smiles=frame.loc[i,'smiles'], error='Invalid chemical-space SMILES') for i in frame.index[~valid]]
    clean = frame.loc[valid].reset_index(drop=True)
    clean['dataset_label'] = 'candidate' if bundle else 'reference'
    if clean.empty:
        return None, errors
    mols, systems, selected = geometry.make_3d_molecules(clean, 42, Path(out)/'generated_conformers.sdf')
    selected_set = set(selected)
    errors.extend(dict(row=int(clean.iloc[i]['_row']), smiles=clean.iloc[i].smiles,
                       error='Conformer generation failed') for i in range(len(clean)) if i not in selected_set)
    clean = clean.iloc[selected].reset_index(drop=True)
    species = sorted({s for system in systems for s in system.get_chemical_symbols()}) if bundle is None else bundle['species']
    supported = [set(s.get_chemical_symbols()).issubset(species) for s in systems]
    if not all(supported):
        for i, ok in enumerate(supported):
            if not ok: errors.append(dict(row=int(clean.iloc[i]['_row']), smiles=clean.iloc[i].smiles, error='Element outside reference SOAP species'))
        mols = [m for m, ok in zip(mols, supported) if ok]
        systems = [s for s, ok in zip(systems, supported) if ok]
        clean = clean.loc[supported].reset_index(drop=True)
    if clean.empty: return None, errors
    soap = SOAP(species=species, periodic=False, r_cut=5., n_max=4, l_max=3, average='inner', sparse=False)
    structural = np.asarray(soap.create(systems), dtype=np.float32)
    names = [name for name, _ in Descriptors._descList] if bundle is None else bundle['descriptor_names']
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(names)
    matrix = np.asarray([calculator.CalcDescriptors(Chem.RemoveHs(m)) for m in mols], dtype=float)
    matrix[~np.isfinite(matrix)] = np.nan
    if bundle is None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            medians = np.nanmedian(matrix, axis=0)
        medians[~np.isfinite(medians)] = 0.
    else:
        medians = bundle['medians']
    row, col = np.where(np.isnan(matrix))
    matrix[row, col] = medians[col]
    selected = np.std(matrix, axis=0)>0 if bundle is None else bundle['descriptor_mask']
    if not np.isfinite(structural).all(): raise ValueError('Non-finite SOAP descriptors')
    spec = dict(species=species, descriptor_names=names, medians=medians, descriptor_mask=selected)
    return (clean, matrix[:,selected], structural, spec), errors


def coordinate_table(frame, coords, components, labels):
    result = frame.drop(columns=['_row'], errors='ignore').copy()
    for j in range(components):
        result[f'Bonding_PC_{j+1}'] = coords[:,j]
        result[f'Structural_PC_{j+1}'] = coords[:,components+j]
    result['PCA_1'] = coords[:,0]
    result['PCA_2'] = coords[:,components]
    result['cluster'] = labels
    return result


def save_plot(reference, candidates, output):
    import plotly.graph_objects as go
    fig = go.Figure()
    for cluster, group in reference.groupby('cluster'):
        fig.add_trace(go.Scatter3d(x=group.PCA_1, y=group.PCA_2, z=group.log_P_upconversion,
                                  mode='markers', name=f'Reference {cluster}', text=group.smiles,
                                  marker=dict(size=2, opacity=.35)))
    if candidates is not None and len(candidates):
        fig.add_trace(go.Scatter3d(x=candidates.PCA_1, y=candidates.PCA_2, z=candidates.log_P_upconversion,
                                  mode='markers', name='FragNet candidates', text=candidates.smiles,
                                  marker=dict(size=7, color='red', symbol='diamond')))
    fig.update_layout(scene=dict(xaxis_title='Bonding PC1', yaxis_title='Structural PC1', zaxis_title='log10 P_upconversion'),
                      title='Fixed reference space: measured/computed reference targets and FragNet candidate predictions')
    fig.write_html(str(output), include_plotlyjs=True)


def fit_reference(frame, out, components=3, weight=1., threshold=.5, clusters=20):
    out = Path(out)
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'Choose a new reference directory: {out}')
    out.mkdir(parents=True, exist_ok=True)
    described, errors = describe(frame, out)
    pd.DataFrame(errors, columns=['row','smiles','error']).to_csv(out/'rejected.csv', index=False)
    if described is None: raise ValueError('No reference descriptors')
    clean, bonding, structural, spec = described
    bundle, coords, labels = fit_coordinates(bonding, structural, clean.log_P_upconversion.to_numpy(float), components, weight, threshold, clusters)
    bundle.update(spec)
    from rdkit import rdBase
    import sklearn, importlib.metadata
    bundle['versions'] = dict(rdkit=rdBase.rdkitVersion, sklearn=sklearn.__version__, dscribe=importlib.metadata.version('dscribe'))
    bundle['reference'] = coordinate_table(clean, coords, components, labels)
    bundle['reference'].to_csv(out/'reference.csv', index=False)
    with (out/'reference.pkl').open('wb') as f: pickle.dump(bundle, f, pickle.HIGHEST_PROTOCOL)
    save_plot(bundle['reference'], None, out/'reference.html')
    info = dict(n_reference=len(clean), components_per_descriptor=components, clustering_dimensions=components*2+1,
                property='log_P_upconversion', property_weight=weight, birch_threshold=threshold,
                birch_subclusters=len(bundle['centres']), final_clusters=clusters, versions=bundle['versions'],
                candidate_assignment='nearest BIRCH subcluster mapped to fixed Ward cluster',
                reference_policy='unique canonical Au-containing identities; mean of distinct target values; descriptor map removes literal [Au] as in dim4',
                legacy_coordinates_preserved=False)
    (out/'reference.json').write_text(json.dumps(info, indent=2)+'\n')
    return info


def fit_existing(cache_dir, embedding_csv, out, components=3, weight=1., threshold=.5,
                 clusters=20, clustered_csv=None):
    """Reuse dim4 matrices and saved coordinates; never regenerate reference descriptors."""
    from rdkit import Chem, rdBase
    import sklearn, importlib.metadata
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import Birch, AgglomerativeClustering
    from sklearn.metrics import pairwise_distances_argmin_min
    out = Path(out); cache_dir = Path(cache_dir)
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'Choose a new reference directory: {out}')
    metadata = json.loads((cache_dir/'metadata.json').read_text())
    expected = {'soap_r_cut':5., 'soap_n_max':4, 'soap_l_max':3, 'soap_average':'inner'}
    for key, value in expected.items():
        if metadata.get(key) != value: raise ValueError(f'Legacy descriptor policy changed: {key}')
    frame = pd.read_csv(embedding_csv, low_memory=False).reset_index(drop=True)
    bonding = np.load(cache_dir/'bonding_descriptor_matrix.npy', mmap_mode='r')
    structural = np.load(cache_dir/'soap_structural_matrix.npy', mmap_mode='r')
    names = np.load(cache_dir/'bonding_descriptor_names.npy', allow_pickle=False).astype(str).tolist()
    valid_indices = np.load(cache_dir/'valid_indices.npy', allow_pickle=False)
    if len(frame) != len(bonding) or len(frame) != len(structural) or len(frame) != len(valid_indices):
        raise ValueError('Cached matrices and saved embedding have different row counts')
    if len(frame) < clusters or len(names) != bonding.shape[1]:
        raise ValueError('Insufficient rows or descriptor names in cache')
    if not {'smiles','log_P_upconversion','chemplot_smiles'}.issubset(frame):
        raise ValueError('Saved embedding needs smiles, chemplot_smiles, log_P_upconversion')
    original_components = 0
    while f'Bonding_PC_{original_components+1}' in frame and f'Structural_PC_{original_components+1}' in frame:
        original_components += 1
    if original_components < components:
        raise ValueError(f'Saved embedding has only {original_components} PCs per descriptor')
    # Recover PCA from all rows; only clustering filters missing properties.
    target = pd.to_numeric(frame.log_P_upconversion, errors='coerce').to_numpy(float)
    finite_target = np.isfinite(target)
    usable_count = int(finite_target.sum())
    if usable_count < clusters:
        raise ValueError(f'Only {usable_count} molecules have finite log P; need at least {clusters}')
    # Keep the original component count for randomized PCA recovery.
    transforms = {}
    for name, matrix in [('bonding',bonding), ('structural',structural)]:
        scaler = StandardScaler().fit(matrix)
        pca = PCA(n_components=original_components, svd_solver='randomized',
                  random_state=int(metadata['random_state']))
        fit_pc = pca.fit_transform(scaler.transform(matrix))
        stored = frame[[f'{name.capitalize()}_PC_{i}' for i in range(1,original_components+1)]].to_numpy(float)
        if not np.allclose(fit_pc[:,:components], stored[:,:components], rtol=1e-3, atol=1e-3):
            raise ValueError(f'{name} cache does not reproduce saved PCA coordinates; check cache/CSV pair')
        # Align transform (X@V) with the saved fit_transform (U*S) coordinates.
        transformed = pca.transform(scaler.transform(matrix))
        source, target_pc = transformed[:,:components], stored[:,:components]
        source_mean, target_mean = source.mean(axis=0), target_pc.mean(axis=0)
        source_var = np.sum((source-source_mean)**2,axis=0)
        if (source_var <= 0).any(): raise ValueError(f'{name} has constant projected PCs')
        coordinate_scale = np.sum((source-source_mean)*(target_pc-target_mean),axis=0)/source_var
        coordinate_offset = target_mean-coordinate_scale*source_mean
        adjusted = source*coordinate_scale+coordinate_offset
        sd = target_pc.std(axis=0)
        relative_rmse = np.sqrt(np.mean((adjusted-target_pc)**2,axis=0))/sd
        print(f'{name} PCA projection alignment relative RMSE: {relative_rmse.tolist()}',flush=True)
        if not np.isfinite(relative_rmse).all() or (relative_rmse > 0.01).any():
            raise ValueError(f'{name} projected PCs differ by over 1% of the saved PC standard deviation; refusing uncertain placement')
        transforms[name] = dict(scaler=scaler,pca=pca,
                                coordinate_scale=coordinate_scale, coordinate_offset=coordinate_offset)
    all_frame = frame
    frame = all_frame.loc[finite_target].reset_index(drop=True)
    frame['log_P_upconversion'] = target[finite_target]
    coords = np.column_stack([frame[[f'Bonding_PC_{i}' for i in range(1,components+1)]].to_numpy(float),
                              frame[[f'Structural_PC_{i}' for i in range(1,components+1)]].to_numpy(float),
                              frame.log_P_upconversion.to_numpy(float)])
    mean = coords.mean(axis=0)
    scales = np.r_[np.repeat(coords[:,0].std(), components),
                   np.repeat(coords[:,components].std(), components),coords[:,-1].std()/weight]
    if not np.isfinite(scales).all() or (scales <= 0).any(): raise ValueError('Invalid weight or constant reference axes')
    matrix = (coords-mean)/scales
    birch = Birch(threshold=threshold,n_clusters=None).fit(matrix)
    centres = birch.subcluster_centers_
    if len(centres)<clusters: raise ValueError('Too few BIRCH subclusters; lower threshold')
    ward = AgglomerativeClustering(n_clusters=clusters,linkage='ward').fit_predict(centres)
    sub, _ = pairwise_distances_argmin_min(matrix,centres)
    labels = ward[sub]
    assignment = 'nearest_birch_subcluster'
    if clustered_csv is not None:
        old = pd.read_csv(clustered_csv).reset_index(drop=True)
        required = ['smiles','BIRCH_Agglomerative_cluster', 'Bonding_PC_1','Structural_PC_1']
        if len(old) == len(all_frame):
            old = old.loc[finite_target].reset_index(drop=True)
        if any(c not in old for c in required) or len(old)!=len(frame):
            raise ValueError('Existing analysis must contain all finite-target reference rows in the same order')
        if not old.smiles.fillna('').equals(frame.smiles.fillna('')):
            raise ValueError('Existing analysis molecules are not aligned with embedding rows')
        for col in ('Bonding_PC_1','Structural_PC_1'):
            if not np.allclose(old[col],frame[col],rtol=1e-6,atol=1e-6):
                raise ValueError(f'Existing analysis coordinates do not match: {col}')
        labels = old.BIRCH_Agglomerative_cluster.to_numpy()
        assignment = 'nearest_reference'
    species = {'H'}
    for smi in all_frame.chemplot_smiles.dropna():
        mol = Chem.MolFromSmiles(smi)
        if mol is None: raise ValueError(f'Invalid saved embedding SMILES: {smi}')
        species.update(atom.GetSymbol() for atom in mol.GetAtoms())
    if 'atomic_numbers' in all_frame:
        for entry in all_frame.atomic_numbers.dropna():
            species.update(Chem.GetPeriodicTable().GetElementSymbol(int(i)) for i in json.loads(entry) if int(i)!=79)
    descriptor_list = [name for name,_ in __import__('rdkit.Chem.Descriptors',fromlist=['_descList'])._descList]
    if not set(names).issubset(descriptor_list): raise ValueError('RDKit descriptor names differ from legacy cache')
    with np.errstate(all='ignore'):
        medians = np.nanmedian(np.asarray(bonding),axis=0)
    medians[~np.isfinite(medians)] = 0.
    mask = np.asarray([name in names for name in descriptor_list],dtype=bool)
    bundle = dict(components=components, weight=weight, bonding=transforms['bonding'],
                  structural=transforms['structural'], mean=mean,scales=scales,
                  centres=centres,subcluster_labels=ward,reference_matrix=matrix,
                  descriptor_names=descriptor_list, descriptor_mask=mask,
                  medians=medians, species=sorted(species), assignment=assignment,
                  versions=dict(rdkit=rdBase.rdkitVersion, sklearn=sklearn.__version__,
                                dscribe=importlib.metadata.version('dscribe')))
    # Candidate imputation indexes all descriptors, including discarded ones.
    full_medians = np.zeros(len(descriptor_list)); full_medians[mask] = medians
    bundle['medians']=full_medians
    bundle['reference']=coordinate_table(frame,coords,components,labels)
    out.mkdir(parents=True,exist_ok=True)
    bundle['reference'].to_csv(out/'reference.csv',index=False)
    with (out/'reference.pkl').open('wb') as f: pickle.dump(bundle,f,pickle.HIGHEST_PROTOCOL)
    save_plot(bundle['reference'],None,out/'reference.html')
    info = dict(n_reference=len(frame), n_pca_rows=len(all_frame),
                rows_without_finite_log_P=len(all_frame)-len(frame), reused_cache=str(cache_dir.resolve()),
                reused_embedding=str(Path(embedding_csv).resolve()),
                descriptor_recalculation=False, components_per_descriptor=components,
                property='log_P_upconversion', property_weight=weight,
                birch_subclusters=len(centres), final_clusters=len(set(labels)),
                original_clusters_preserved=clustered_csv is not None,
                candidate_assignment=assignment)
    (out/'reference.json').write_text(json.dumps(info,indent=2)+'\n')
    return info


def project(frame, reference_dir, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    with (Path(reference_dir)/'reference.pkl').open('rb') as f: bundle = pickle.load(f)
    from rdkit import rdBase
    import sklearn, importlib.metadata
    versions = dict(rdkit=rdBase.rdkitVersion, sklearn=sklearn.__version__, dscribe=importlib.metadata.version('dscribe'))
    if versions != bundle['versions']: raise RuntimeError('Use the same RDKit/sklearn/DScribe versions as the reference fit')
    described, errors = describe(frame, out, bundle)
    pd.DataFrame(errors, columns=['row','smiles','error']).to_csv(out/'projection_rejected.csv', index=False)
    if described is None: raise ValueError('No candidates could be projected')
    clean, bonding, structural, _ = described
    coords, labels, distances, nearest, ref_distances = transform_coordinates(bundle, bonding, structural, clean.log_P_upconversion.to_numpy(float))
    result = coordinate_table(clean, coords, bundle['components'], labels)
    result['distance_to_birch_subcluster'] = distances
    result['nearest_reference_distance'] = ref_distances
    result['nearest_reference_smiles'] = bundle['reference'].iloc[nearest].smiles.to_numpy()
    result['nearest_reference_log_P_upconversion'] = bundle['reference'].iloc[nearest].log_P_upconversion.to_numpy()
    result.to_csv(out/'candidates_in_space.csv', index=False)
    save_plot(bundle['reference'], result, out/'candidates_in_space.html')
    return result
