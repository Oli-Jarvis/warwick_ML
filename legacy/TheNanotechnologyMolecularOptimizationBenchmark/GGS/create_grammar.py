import numpy as np
from group_selfies import Group as selfies_group
from group_selfies import GroupGrammar
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from GGS.encoding import disable_rdkit_logging
import json
from io import BytesIO
from PIL import ImageDraw
disable_rdkit_logging()

def test():
    g = selfies_group('toluene', 'CC1=CC=CC=C1', all_attachment=True)
    g_1 = selfies_group('benzene', 'C1=CC=CC=C1', all_attachment=True)
    g_2 = selfies_group('naphthalene', 'C1=CC=C2C=CC=CC2=C1', all_attachment=True)
    g_3 = selfies_group('anthracene', 'C1=CC=C2C=C3C=CC=CC3=CC2=C1', all_attachment=True)
    g_4 = selfies_group('ethane', 'CC', all_attachment=True)
    g_5 = selfies_group('ethylene', 'C=C', all_attachment=True)
    grammar = GroupGrammar([g, g_1, g_2, g_3, g_4, g_5])  # creating a grammar using a list
    print(grammar.vocab)
    grammar.to_file("./data/test.txt")

def create_grammar(json_path, out_path, grammar_name, filter_S = False, filter_charge = False):
    """
    Create a grammar from the PDB 105 dataset and save it to a file.
    """

    with open(json_path, 'r') as f:
        data = json.load(f)

    #get values from keys block_name, block_smi
    block_names = [data['block_name'][str(i)] for i in range(len(data['block_name']))]
    print(block_names)
    block_smiles = [data['block_smi'][str(i)] for i in range(len(data['block_smi']))]

    groups = []
    mols = []
    failed_counter = 0
    smis_recreated = []

    for i in range(len(block_names)):
        #print(f"Block {i}: Name: frag_{i}, SMILES: {block_smiles[i]}, R: {block_r[i]}")
        try:
            print(block_smiles[i])
            mol = Chem.MolFromSmiles(block_smiles[i])
            if not mol:
                raise ValueError("Invalid SMILES string provided.")

            # Add explicit hydrogens
            mol_with_hs = Chem.AddHs(mol)

            #check if filter_S is True, then skip molecules with sulfur
            if filter_S:
                has_sulfur = any(atom.GetAtomicNum() == 16 for atom in mol_with_hs.GetAtoms())
                if has_sulfur:
                    print(f"Skipping molecule with sulfur: {block_smiles[i]}")
                    continue

            if filter_charge:
                formal_charge = Chem.GetFormalCharge(mol_with_hs)
                if formal_charge != 0:
                    print(f"Skipping charged molecule: {block_smiles[i]} with charge {formal_charge}")
                    continue


            em = Chem.EditableMol(mol_with_hs)
            h_indices_to_replace = []
            core_atoms_checked = set()
            for atom in mol_with_hs.GetAtoms():
                if atom.GetAtomicNum() == 1:
                    #get the neighbor atom
                    neighbor = atom.GetNeighbors()
                    assert len(neighbor) == 1
                    if neighbor[0].GetIdx() in core_atoms_checked:
                        continue
                    core_atoms_checked.add(neighbor[0].GetIdx())
                    h_indices_to_replace.append(atom.GetIdx())

            for h_idx in sorted(h_indices_to_replace, reverse=True):
                # Create a new dummy atom (*)
                dummy_atom = Chem.Atom(0)
                # Set its atom map number. This is the key change.
                map_number = 1
                dummy_atom.SetAtomMapNum(map_number)
                # Replace the hydrogen atom at the specific index
                em.ReplaceAtom(h_idx, dummy_atom)

            final_mol = em.GetMol()
            #remove explict hydrogens
            final_mol = Chem.RemoveHs(final_mol)

            smi_recreated = Chem.MolToSmiles(final_mol)

            group = selfies_group(f"frag_{i}", smi_recreated)
            print(group)

        except Exception as e:
            print(f"Error creating group for block {i} with smi {block_smiles[i]}: {e}")
            raise ValueError(e)
            failed_counter += 1
            continue
        if smi_recreated in smis_recreated:
            print(f"Duplicate SMILES found for block {i}: {smi_recreated}")
            continue
        mols.append(mol)
        smis_recreated.append(smi_recreated)
        groups.append(group)

    grammar = GroupGrammar(groups)
    len_attachment_points = [len(g.attachment_points) for g in groups]
    print(f"Grammar vocabulary size: {len(grammar.vocab)}")
    print(f"Failed to create {failed_counter} groups out of {len(block_names)} blocks.")
    #grammar.to_file(out_path)
    with open(f"{out_path}/{grammar_name}.txt", 'w') as f:
        for i, group in enumerate(groups):
            f.write(f"frag_{i} {smis_recreated[i]} 0\n")

    valid = []
    key_list = list(grammar.vocab.keys())
    for i, key in enumerate(key_list):
        group = grammar.vocab[key]

        try:
            selfies_group(group.name, group.canonsmiles)
            valid.append(group.mol)
            # print(i, group.name, group.canonsmiles)
        except Exception as e:
            #print("here", i, group.name, group.canonsmiles, e)
            continue

    fig = DrawMolsZoomed(valid, molsPerRow=10)
    fig.save(f"{out_path}/{grammar_name}.pdf")


def DrawMolsZoomed(mols, molsPerRow=10, subImgSize=(200, 200)):

    #sort mols by number of atoms keep old indices and sort together with indices
    mols, old_indices = zip(*sorted([(mol, i) for i, mol in enumerate(mols)], key=lambda x: x[0].GetNumAtoms()))

    from PIL import Image
    nRows = len(mols) // molsPerRow
    if len(mols) % molsPerRow: nRows += 1
    fullSize = (molsPerRow * subImgSize[0], nRows * subImgSize[1])
    full_image = Image.new('RGBA', fullSize )
    for ii, mol in enumerate(mols):
        column = ii % molsPerRow
        row = ii // molsPerRow
        offset = ( column*subImgSize[0], row * subImgSize[1] )
        sub = draw_atom_idx(mol, size=subImgSize)
        draw = ImageDraw.Draw(sub)
        draw.text((5, 5), str(old_indices[ii]), fill='black')
        full_image.paste(sub, box=offset)
    return full_image

def draw_atom_idx(Sm, size=(200, 200)):
    from PIL import Image
    Sm = mol_with_atom_index(Sm)
    AllChem.Compute2DCoords(Sm)
    X = rdMolDraw2D.MolDraw2DCairo(*size)
    X.DrawMolecule(Sm)
    X.FinishDrawing()
    return Image.open(BytesIO(X.GetDrawingText()))


def mol_with_atom_index(mol):
    mol_c = Chem.Mol(mol)
    for atom in mol_c.GetAtoms():
        atom.SetProp("atomNote", str(atom.GetIdx()))
    return mol_c





if __name__ == "__main__":
    import os
    data_dir = os.path.join(os.path.dirname(__file__), "..", "genetic_GFN_framework", "data")
    #test()
    create_grammar(os.path.join(data_dir, "blocks_PDB_105_enhanced.json"), data_dir, "GS_complex_grammar")
    create_grammar(os.path.join(data_dir, "blocks_PDB_105_enhanced.json"), data_dir, "GS_complex_grammar_without_S", filter_S=True, filter_charge=True)






