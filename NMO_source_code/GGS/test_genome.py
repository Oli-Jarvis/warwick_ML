import pytest
import selfies as sf
from group_selfies import GroupGrammar
from rdkit import Chem
from rdkit.Chem import Draw

from .ggs import GGS

def test_Group():
    with pytest.raises(ValueError):
        grammar_path = "../data/test_grammar.txt"
        ggs = GGS("[:0toluene][Ring2][:0trifluoromethane][:0sulfonamide]", grammar_path=grammar_path)
        
    ggs = GGS("[:0toluene][Ring2][:2pyrazole][Ring2][:0trifluoromethane][pop][Branch][:0sulfonamide]", grammar_path=grammar_path)
    assert len(ggs.groups) == 4
    #assert that the ggs._selfies_graph has 4 nodes
    assert len(ggs._selfies_graph) == 4
    #assert that the ggs._selfies_graph has 3 edges
    assert len(ggs._selfies_graph.edges) == 3

    #assert stuff that node 2 (<Group pyrazole N1=C(*1)-C(*1)=C(*1)-N-1*1>)
    assert len(ggs._selfies_graph.adj[ggs.groups[2]]) == 2
    assert ggs.groups[2].name == 'pyrazole'
    dict_to_check = ggs._selfies_graph.adj[ggs.groups[2]][ggs.groups[1]]
    assert ggs.groups[1].name == 'trifluoromethane'
    assert dict_to_check['start_attachment_point'] == 0
    assert dict_to_check['end_attachment_point'] == 0

    dict_to_check = ggs._selfies_graph.adj[ggs.groups[2]][ggs.groups[3]]
    assert ggs.groups[3].name == 'sulfonamide'
    assert dict_to_check['start_attachment_point'] == 3
    assert dict_to_check['end_attachment_point'] == 0

    with pytest.raises(KeyError):
        ggs._selfies_graph.adj[ggs.groups[2]][ggs.groups[0]]

    #get number of atoms in the ggs
    assert ggs.get_num_atoms() == 20

    #grammar_path = "./src/problem_specification/test_grammar.txt"
    grammar = GroupGrammar.from_file(grammar_path)
    valid = grammar.decoder(ggs.encoding)
    valid = Chem.Mol(valid)
    #plot the molecule
    #img = Draw.MolToImage(valid)
    #img.show()
    num_atoms = valid.GetNumAtoms()
    assert num_atoms == ggs.get_num_atoms() 
    
    categorized_grammar = ggs.categorize_grammar()
    assert len(categorized_grammar[0]) == 1
    assert len(categorized_grammar[1]) == 1
    assert len(categorized_grammar[2]) == 3

    
    
if __name__ == "__main__":
    pass


