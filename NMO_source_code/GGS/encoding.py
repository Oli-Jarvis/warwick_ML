import copy
import random
import numpy as np

from group_selfies import constants as gs_const, GroupGrammar, Group as selfies_group
import re
from networkx.algorithms.shortest_paths.unweighted import predecessor

try:
    from .group import Group
except ImportError:
    from group import Group
import networkx as nx

from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt



class GGS:
    def __init__(self, encoding=None, grammar=None, rng=None):
        """
        Initialize the GGS object with a group selfies encoding and a
        PRE-LOADED grammar object.

        Args:
            encoding: The group selfies string (optional).
            grammar: A pre-loaded GroupGrammar object (REQUIRED).
            rng: Random number generator.
        """
        if grammar is None:
            raise ValueError("GGS __init__ requires a pre-loaded 'grammar' object.")

        self.encoding = encoding
        self.grammar = grammar
        self.grammar_path = getattr(grammar, 'grammar_path', None)

        self._categorized_grammar = None
        self._grouped_selfies_list = None
        self._mol = None
        self._anchors = None
        if encoding is not None:
            self.encoding_to_graph()

        self.rng = rng if rng is not None else np.random.default_rng()
        disable_rdkit_logging()

    @classmethod
    def from_grammar_path(cls, encoding=None, grammar_path='test_grammar.txt', rng=None):
        """
        Class method to initialize a GGS by loading the grammar from a file path.
        This is slower and should only be used for testing or one-off instances.

        Args:
            encoding:
            grammar_path:
            rng: Random number generator
        """

        grammar = GroupGrammar.from_file(grammar_path)

        # Store the path on the grammar object so the __init__ can find it
        grammar.grammar_path = grammar_path

        # Call the main, fast __init__ with the loaded grammar object
        return cls(encoding=encoding, grammar=grammar, rng=rng)


    @property
    def categorized_grammar(self):
        if self._categorized_grammar is None:
            self._categorized_grammar = self.categorize_grammar()
        return self._categorized_grammar


    @categorized_grammar.setter
    def categorized_grammar(self, value):
        self._categorized_grammar = value

    @property
    def grouped_selfies_list(self):
        if self._grouped_selfies_list is None and self.encoding is not None:
            self._grouped_selfies_list = self.split_group_selfies(self.encoding)
        return self._grouped_selfies_list

    @property
    def mol(self):
        if self._mol is not None:
            return self._mol
        if self.encoding is None:
            raise ValueError("No encoding available")
        if self._selfies_graph is None:
            raise ValueError("No graph available")
        if self._anchors is None and self._mol is None:
            mol, anchors = self.__get_mol_info()
            self._mol = mol
            self._anchors = anchors
        return self._mol

    @property
    def anchors(self):
        """
        Get the indices of the anchor positions in the molecule.
        """
        if self._anchors is not None:
            return self._anchors
        if self.encoding is None:
            raise ValueError("No encoding available")
        if self._selfies_graph is None:
            raise ValueError("No graph available")
        if self._mol is None and self._anchors is None:
            mol, anchors = self.__get_mol_info()
            self._anchors = anchors
            self._mol = mol
        return self._anchors

    def __get_mol_info(self, draw_steps=False):
        """
        Get the rdkit mol object and the anchor positions. Explicit attachment points are removed.
        :param draw_steps: Draw the molecule after each step
        :return: mol, [anchor_l,anchor_r]
        """
        #draw_steps = True
        if self.encoding is None:
            raise ValueError("No encoding available")
        if self._selfies_graph is None:
            raise ValueError("No graph available")

        #stuff for lef anchor
        #get start node from graph
        start_node = [node for node in self._selfies_graph.nodes if self._selfies_graph.in_degree(node) == 0]
        assert len(start_node) == 1
        start_node = start_node[0]
        anchor_l = start_node.attachment_points[start_node.occupied_attachment_points[0]]
        anchor_l = start_node.parent_map[anchor_l]
        anchor_r = -1

        def mol_without_attachment_points(mol):
            to_remove = []
            new_mol = Chem.RWMol(mol)
            for idx, atom in enumerate(new_mol.GetAtoms()):
                if atom.GetSymbol() == '*':
                    to_remove.append(idx)
            for idx in reversed(to_remove):
                new_mol.RemoveAtom(idx)
            return new_mol

        def n_attachment_points_to_index(mol, index):
            counter = 0
            new_mol = Chem.RWMol(mol)
            for idx, atom in enumerate(new_mol.GetAtoms()):
                if idx == index:
                    return counter
                if atom.GetSymbol() == '*':
                    counter += 1
            return counter


        def dfs(node, start_index):
            nonlocal mol, visited, anchor_r

            visited.add(node)
            node.index_in_mol = start_index
            if start_index > mol.GetNumAtoms():
                raise ValueError("Something wrong here")


            if node.reserved_attachment_point != -1:
                #print("reserved attachment point found", node.reserved_attachment_point)
                anchor_r = node.reserved_attachment_point
                anchor_r = node.attachment_points[anchor_r]
                anchor_r = node.index_in_mol + node.parent_map[anchor_r]
                #print("found right anchor", anchor_r)

            for neighbor in self._selfies_graph.neighbors(node):
                if neighbor not in visited:

                    start = self._selfies_graph.edges[node, neighbor]['start_attachment_point']
                    start = node.index_in_mol + node.parent_map[node.attachment_points[start]]

                    if start> mol.GetNumAtoms():
                        raise ValueError(f"Something wrong here with start {start=}, {mol.GetNumAtoms()=}")

                    mol = Chem.CombineMols(mol, neighbor.mol)
                    mol = Chem.RWMol(mol)

                    end = self._selfies_graph.edges[node, neighbor]['end_attachment_point']
                    N_atoms = mol.GetNumAtoms()
                    end = N_atoms - neighbor.mol.GetNumAtoms() + neighbor.parent_map[neighbor.attachment_points[end]]
                    if draw_steps:
                        #DrawingOptions.includeAtomNumbers=True
                        mol_draw = mol_without_attachment_points(mol)
                        dos = Draw.MolDrawOptions()
                        dos.addAtomIndices = True
                        img = Draw.MolToImage(mol_draw, options=dos)
                        img.show()

                    #add bond
                    mol.AddBond(start, end, Chem.BondType.SINGLE)

                    start_index = mol.GetNumAtoms()-neighbor.mol.GetNumAtoms()
                    dfs(neighbor, start_index)

        visited = set()
        mol = start_node.mol
        mol = Chem.RWMol(mol)
        dfs(start_node, 0)

        #plot
        if draw_steps:
            mol_draw = mol_without_attachment_points(mol)
            dos = Draw.MolDrawOptions()
            dos.addAtomIndices = True
            img = Draw.MolToImage(mol_draw, options=dos, size=(500,500))
            img.show()

        if anchor_r == -1:
            raise ValueError("No right anchor found")

        anchor_l_atom_prev = mol.GetAtomWithIdx(anchor_l)
        anchor_r_atom_prev = mol.GetAtomWithIdx(anchor_r)

        #correct for attachment points
        anchor_l -= n_attachment_points_to_index(mol, anchor_l)
        anchor_r -= n_attachment_points_to_index(mol, anchor_r)
        #print(f"{anchor_l=}, {anchor_r=}")

        #remove attachment points
        mol = mol_without_attachment_points(mol)

        #mark atoms as attachment points
        mol = Chem.RWMol(mol)
        mol.GetAtomWithIdx(anchor_l).SetProp("atomNote", "L")
        mol.GetAtomWithIdx(anchor_r).SetProp("atomNote", "R")

        #see if something went wrong
        anchor_l_atom = mol.GetAtomWithIdx(anchor_l)
        anchor_r_atom = mol.GetAtomWithIdx(anchor_r)
        assert anchor_l_atom_prev.GetSymbol() == anchor_l_atom.GetSymbol()
        assert anchor_l_atom_prev.GetExplicitValence() == anchor_l_atom.GetExplicitValence()
        assert anchor_r_atom_prev.GetSymbol() == anchor_r_atom.GetSymbol()
        assert anchor_r_atom_prev.GetExplicitValence() == anchor_r_atom.GetExplicitValence()


        return mol, [anchor_l, anchor_r]

    def __str__(self):
        # Return a string representation of the object.
        return f"{self.encoding}"

    def __eq__(self, other):
        return self.encoding == other.encoding

    def categorize_grammar(self, grammar=None):
        """
        Categorize the grammar into the different groups with different numbers of attachment points
        :return:[[group names with n=1], [group names with n=2], [group names with n=2]]]
        """
        if grammar is None:
            grammar = self.grammar
        categorized_grammar = [set(),set(),set()]
        for group_name in grammar.vocab:
            group = grammar.vocab[group_name]
            n_attachment_points = len(group.attachment_points)
            if n_attachment_points == 0:
                continue
            #if n_attachment_points > 3
            category = lambda n: n-1 if n < 3 else 2
            categorized_grammar[category(n_attachment_points)].add(group)
        #sort for reproducibility
        sorted_grammar = []
        for category_set in categorized_grammar:
            # Sort by a unique, stable attribute like the group name
            sorted_list = sorted(list(category_set), key=lambda group: group.name)
            sorted_grammar.append(sorted_list)
        return sorted_grammar


    def graph_to_encoding(self, graph=None):
        """
        Convert the graph to a group selfies encoding
        :param graph:
        :return:
        """
        if graph is None:
            graph = self._selfies_graph
        encoding = ""
        start_node = [node for node in graph.nodes if graph.in_degree(node) == 0]
        assert len(start_node) == 1
        start_node = start_node[0]

        anchor_r_found = False
        n_nodes = len(graph.nodes)

        def dfs(node, prev_node, neighbor_id=0):
            nonlocal visited, encoding, anchor_r_found

            visited.add(node)

            neighbours = list(graph.neighbors(node))
            name = node.name
            start_attachment_point = node.attachment_point_start
            #start_attachment_point = node.occupied_attachment_points[0]

            #The following sorting is necessary to avoid double pops. This might not be necessary after mutation.
            #now sort neighbours by the depth. We have to go to the shallow nodes first -> no double pop possible.
            #This assumes that we have side branches with length 1
            length = lambda node: len(list(nx.descendants(graph, node)))
            neighbours = sorted(neighbours, key=length)

            #if we have several neighbours with zero depth, we have to visit the one with a reserved attachment point last
            neighbours_depth = [length(neighbor) for neighbor in neighbours]
            #check if only zero depth and swap if necessary
            if all([depth == 0 for depth in neighbours_depth]):
                has_reserved = [neighbor.reserved_attachment_point != -1 for neighbor in neighbours]
                index_with_reserved = np.where(has_reserved)[0]
                if len(index_with_reserved) == 1:
                    index_with_reserved = index_with_reserved[0]
                    #sort by depth
                    tmp = neighbours[-1]
                    neighbours[-1] = neighbours[index_with_reserved]
                    neighbours[index_with_reserved] = tmp

            # the order of neighbours and occupied attachment points is important. This might be mixed up by mutation stuff
            # reorder it here
            edge_order = [node.attachment_point_start]
            for i, neighbor in enumerate(neighbours):
                edge = graph.edges[node, neighbor]
                edge_start_attachment_point = edge['start_attachment_point']
                edge_order.append(edge_start_attachment_point)
            # now set the occupied attachment points in the right order
            node.occupied_attachment_points = edge_order


            #we are not right at the right anchor node
            if len(neighbours) == 0 and node.reserved_attachment_point == -1:
                index_1 = neighbor_id + 1
                index_2 = index_1 + 1
                index_1 = index_1 % len(prev_node.occupied_attachment_points)
                index_2 = index_2 % len(prev_node.occupied_attachment_points)
                #if we are in the last node we have to handle the prev node different because this is the right
                #anchor point and has therefore a reseverd attachment point
                if(len(visited) == n_nodes and anchor_r_found == False):
                    if(prev_node.reserved_attachment_point == -1):
                        raise ValueError("Something went wrong here.")
                    shift = prev_node.reserved_attachment_point - prev_node.occupied_attachment_points[-1]
                else:
                    shift = prev_node.occupied_attachment_points[index_2] - prev_node.occupied_attachment_points[index_1]

                shift = shift % len(prev_node.attachment_point_occupation)

                string_to_add = f"[:{start_attachment_point}{name}][pop]{gs_const.INDEX_ALPHABET[shift]}"
                encoding += string_to_add

            #we are in the right anchor node
            elif len(neighbours) == 0 and node.reserved_attachment_point != -1:
                shift = node.reserved_attachment_point - start_attachment_point
                shift = shift % len(node.attachment_point_occupation)

                string_to_add = f"[:{start_attachment_point}{name}]{gs_const.INDEX_ALPHABET[shift]}"
                encoding += string_to_add
                anchor_r_found = True


            else:
                name = node.name
                start_attachment_point = node.occupied_attachment_points[0]
                end_attachment_point = node.occupied_attachment_points[1]
                shift = end_attachment_point - start_attachment_point
                shift = shift % len(node.attachment_point_occupation)
                string_to_add = f"[:{start_attachment_point}{name}]{gs_const.INDEX_ALPHABET[shift]}"
                encoding += string_to_add

                for i, neighbor in enumerate(neighbours):
                    if neighbor not in visited:
                        dfs(neighbor, node, i)

        visited = set()
        dfs(start_node, None, 0)

        return encoding

    def encoding_to_graph(self, encoding=None):
        """
        Convert the group selfies encoding to a graph
        :param encoding:
        :return:
        """

        if(encoding is not None):
            self.encoding = encoding

        self._selfies_graph = nx.DiGraph()
        self.groups = []
        for i, part in enumerate(self.grouped_selfies_list):
            if ":" in part[0]:
                part[0] = part[0].replace("[", "").replace("]", "")
                pattern = r':(\d+)(.*)'
                match = re.search(pattern, part[0])

                group_name = match.group(2)
                group = Group(group_name, self.grammar.vocab[group_name].canonsmiles)
                attachment_point_start = int(match.group(1))
                attachment_point_start = group.next_avail_attachment(attachment_point_start)
                if attachment_point_start == -1:
                    raise ValueError(
                        f"No attachment point available in group {group}, {group.occupied_attachment_points}")
                group.occupy_attachment_point(attachment_point_start)

                if (len(part) > 1):
                    attachment_point_end = gs_const.INDEX_CODE[part[1]]
                    attachment_point_end = group.next_avail_attachment(attachment_point_end)
                    if attachment_point_end == -1:
                        raise ValueError(
                            f"No attachment point available in group {group}, {group.attachment_point_occupation} {attachment_point_end}")
                    group.reserve_attachment_point(attachment_point_end)

                self.groups.append(group)

                # update graph
                self._selfies_graph.add_node(group, group=group)
                if (len(self.groups) > 1):
                    if (self.groups[-2].reserved_attachment_point != -1):
                        edge_start_attachment_point = self.groups[-2].reserved_attachment_point
                        assert edge_start_attachment_point != -1
                        self.groups[-2].occupy_attachment_point(edge_start_attachment_point)
                        edge_end_attachment_point = group.occupied_attachment_points[-1]
                        self._selfies_graph.add_edge(self.groups[-2], group,
                                                     start_attachment_point=edge_start_attachment_point,
                                                     end_attachment_point=edge_end_attachment_point)
                    else:
                        raise ValueError(
                            f"No attachment point available in previous group {self.groups[-2]}, {self.groups[-2].attachment_point_occupation}")

            elif "[pop]" in part[0]:
                shift = 0
                if len(part) > 1:
                    shift = gs_const.INDEX_CODE[part[1]]
                # go back to group
                if len(self.groups) < 1:
                    raise ValueError("No group to pop")
                elif len(self.groups) == 1:
                    continue

                # free reserved attachment point in last group
                self.groups[-1].reserved_attachment_point = -1

                back_to = self.groups[-2]
                next_point = back_to.next_avail_attachment(shift)
                if next_point == -1:
                    raise ValueError(
                        f"No attachment point available in group {back_to}, {back_to.occupied_attachment_points}, {back_to.attachment_point_occupation}")

                back_to.reserve_attachment_point(next_point)
                # move back_to group to the end of the list
                self.groups = self.groups[:-2] + [self.groups[-1]] + [self.groups[-2]]

    def create_random_genome(self, n_groups, n_explicit_pops):
        """
        Create a random genome with n_groups groups and n_pops pop tokens
        :param n_groups: Max number of groups
        :param n_explicit_pops: Max number of explicit pop tokens
        :return:
        """

        self._mol = None
        self._anchors = None
        self._selfies_graph = nx.DiGraph()
        encoding = ""
        groups = []

        #first group
        allowed_categories = [1, 2]
        category = self.rng.choice(allowed_categories)
        prev_category = category
        selected_group = self.rng.choice(self.categorized_grammar[category])
        selected_group = Group(selected_group.name, selected_group.canonsmiles)
        groups.append(selected_group)
        attachment_points = self.rng.choice(len(selected_group.attachment_points), 2, replace=False)
        attachment_point_start = attachment_points[0]
        attachment_point_end = attachment_points[1]
        assert attachment_point_start != attachment_point_end
        encoding += f"[:{attachment_point_start}{selected_group.name}]{gs_const.INDEX_ALPHABET[attachment_point_end]}"
        # update graph
        self._selfies_graph.add_node(selected_group, group=selected_group)
        assert selected_group.occupy_attachment_point(attachment_point_start) == True
        attachment_point_end = selected_group.next_avail_attachment(attachment_point_end)
        assert selected_group.reserve_attachment_point(attachment_point_end) == True

        for i in range(n_groups-1):
             #we have to pop or we want to pop
            if prev_category == 0:

                n_attachment_points = len(groups[-2].attachment_points)
                shift = self.rng.choice(n_attachment_points)
                avail_attachment_points = groups[-2].next_avail_attachment(shift)
                
                encoding += f"[pop]{gs_const.INDEX_ALPHABET[shift]}"
                #avoid double pops
                prev_category = 1

                # free reserved attachment point in last group
                groups[-1].reserved_attachment_point = -1

                if avail_attachment_points == -1:
                    raise ValueError(f"Something is wrong with the attachment points of group {groups[-2]}")

                #update group we go back to
                assert groups[-2].reserve_attachment_point(avail_attachment_points) == True

                #move group we go back to to end of list
                groups = groups[:-2] + [groups[-1]] + [groups[-2]]

            elif prev_category == 1:
                allowed_categories = [1, 2]

                category = self.rng.choice(allowed_categories)
                prev_category = category
                selected_group = self.rng.choice(self.categorized_grammar[category])
                selected_group = Group(selected_group.name, selected_group.canonsmiles)
                groups.append(selected_group)
                attachment_points = self.rng.choice(len(selected_group.attachment_points), 2, replace=False)
                attachment_point_start = attachment_points[0]
                shift = attachment_points[1]

                #update group
                assert selected_group.occupy_attachment_point(attachment_point_start) == True
                attachment_point_end = selected_group.next_avail_attachment(shift)
                assert selected_group.reserve_attachment_point(attachment_point_end) == True

                encoding += f"[:{attachment_point_start}{selected_group.name}]{gs_const.INDEX_ALPHABET[shift]}"
                
                #update graph -> edge from last group to new group
                self._selfies_graph.add_node(selected_group, group=selected_group)
                edge_start_attachment_point = groups[-2].reserved_attachment_point
                assert groups[-2].occupy_attachment_point(edge_start_attachment_point) == True
                edge_end_attachment_point = attachment_point_start
                self._selfies_graph.add_edge(groups[-2], selected_group, start_attachment_point=edge_start_attachment_point, end_attachment_point=edge_end_attachment_point)


            elif prev_category == 2:
                #if explicit pops left
                if n_explicit_pops > 0 and len(groups) > 1:
                    if len(groups[-2].get_free_attachment_points()) > 1:
                        #decide randomly if we pop or not
                        pop = self.rng.choice([True, False], p=[0.99, 0.01])
                        if pop:
                            n_explicit_pops -= 1

                            #free reserved attachment point
                            groups[-1].reserved_attachment_point = -1

                            #determine where it goes on in last group
                            n_attachment_points = len(groups[-2].attachment_points)
                            shift = self.rng.choice(n_attachment_points)
                            attachment_point = groups[-2].next_avail_attachment(shift)
                            if attachment_point == -1:
                                raise ValueError(f"No attachment point available in group {groups[-2]}")
                            assert groups[-2].reserve_attachment_point(attachment_point) == True
                            encoding += f"[pop]{gs_const.INDEX_ALPHABET[shift]}"

                            # move group we go back to end of list
                            groups = groups[:-2] + [groups[-1]] + [groups[-2]]


                #category 0 only allowed if we have more than 2 attachment points left and we are not at the end
                if len(groups[-1].get_free_attachment_points()) > 2 and i != n_groups-2:
                    allowed_categories = [0, 1, 2]
                else:
                    allowed_categories = [1, 2]
                    #allowed_categories = [2]

                category = self.rng.choice(allowed_categories)
                prev_category = category
                selected_group = self.rng.choice(self.categorized_grammar[category])
                selected_group = Group(selected_group.name, selected_group.canonsmiles)
                groups.append(selected_group)
                if category == 0:
                    attachment_point = 0
                    encoding += f"[:{attachment_point}{selected_group.name}]"

                    # update group
                    assert selected_group.occupy_attachment_point(attachment_point) == True

                    # update graph
                    self._selfies_graph.add_node(selected_group, group=selected_group)
                    edge_start_attachment_point = groups[-2].reserved_attachment_point
                    assert groups[-2].occupy_attachment_point(edge_start_attachment_point) == True

                    self._selfies_graph.add_edge(groups[-2], groups[-1], start_attachment_point=edge_start_attachment_point, end_attachment_point=attachment_point)

                else:
                    attachment_points = self.rng.choice(len(selected_group.attachment_points), 2, replace=False)
                    attachment_point_start = attachment_points[0]
                    shift = attachment_points[1]

                    # update group
                    assert selected_group.occupy_attachment_point(attachment_point_start) == True
                    attachment_point_end = selected_group.next_avail_attachment(shift)
                    assert selected_group.reserve_attachment_point(attachment_point_end) == True

                    encoding += f"[:{attachment_point_start}{selected_group.name}]{gs_const.INDEX_ALPHABET[shift]}"

                    #update graph
                    self._selfies_graph.add_node(selected_group, group=selected_group)
                    edge_start_attachment_point = groups[-2].reserved_attachment_point
                    groups[-2].occupy_attachment_point(edge_start_attachment_point)
                    edge_end_attachment_point = attachment_point_start
                    self._selfies_graph.add_edge(groups[-2], selected_group, start_attachment_point=edge_start_attachment_point, end_attachment_point=edge_end_attachment_point)


        self.encoding = encoding
        self.groups = groups
        return encoding


    def get_num_atoms(self):
        """
        Get the number of atoms in the genome graph
        :return:
        """
        n_atoms = 0
        for group in self.groups:
            n_atoms += group.group_size
        return n_atoms


    def get_rdkit_mol(self):
        """
        Get the mol object of the genome. Might differ from the mol object of the graph.
        :return: Molecule object (rdkit)
        """
        gram = GroupGrammar(self.groups)
        if self.encoding is None:
            raise ValueError("No encoding available")
        mol = gram.decoder(self.encoding)

        mol = Chem.Mol(mol)
        #assert mol.GetNumAtoms() == self.get_num_atoms()
        return mol

    def print_graph(self, groups=None, selfies_graph=None):
        #todo: does this have to be a class method?
        if(groups is None):
            groups = self.groups
        if(selfies_graph is None):
            selfies_graph = self._selfies_graph
        for i, group1 in enumerate(groups):
            for j, group2 in enumerate(groups):
                try:
                    dict_to_check = selfies_graph.adj[groups[i]][groups[j]]
                    start = dict_to_check['start_attachment_point']
                    end = dict_to_check['end_attachment_point']
                    #print(f"{groups[i].name}->{groups[j].name} {start=}, {end=} ")
                except KeyError:
                    pass

    def check_only_one_has_reserved(self):
        has_reserved = 0
        has_reserved_g = []
        for i, g in enumerate(self.groups):
            if g.reserved_attachment_point != -1:
                has_reserved += 1
                has_reserved_g.append(g)

        return has_reserved == 1


    def split_group_selfies(self, group_selfies):
        """
        Split in reasonable groups assuming that the input does not contain explicit selfies.
        :param group_selfies:
        :return: List of parts belonging to the fragment
        """

        group_selfies = re.findall(r'\[.*?\]', group_selfies)

        grouped_selfies_list = []
        list_ = []
        #Split in reasoanble groups assuming that the input does not contain explicit selfies
        for i, group in enumerate(group_selfies):
            if ":" in group or "[pop]" in group:
                if(len(list_) > 0):
                    grouped_selfies_list.append(list_)
                list_ = []

            list_.append(group)
            
        grouped_selfies_list.append(list_)



        return grouped_selfies_list

    def create_3d_structure(self, save_xyz=False, save_path=None, anchor_mode = "thiol"):
        """
        Creates a 3D structure of the molecule. Including the anchors. The molecule is saved as a .xyz file. The
        indices of the anchors are returned and written to the header of the xyz-file.
        :return: mol, [anchor_l, anchor_r]
        """

        mol = self.mol
        mol.UpdatePropertyCache(strict=False)
        anchors = self.anchors

        max_valence = {
            'H': 1,
            'He': 0,
            'Li': 1,
            'Be': 2,
            'B': 3,
            'C': 4,
            'N': 3,
            'O': 2,
            'F': 1,
            'Ne': 0,
            'Na': 1,
            'Mg': 2,
            'Al': 3,
            'Si': 4,
            'P': 5,
            'S': 6,
            'Cl': 1,
            'Ar': 0,
            'K': 1,
            'Ca': 2,
            # Add more elements as needed
        }

        anchor_l_atom = mol.GetAtomWithIdx(anchors[0])
        anchor_l_atom_symbol = anchor_l_atom.GetSymbol()
        anchor_r_atom = mol.GetAtomWithIdx(anchors[1])
        anchor_r_atom_symbol = anchor_r_atom.GetSymbol()

        #fix valence of anchor atoms
        if(anchors[0] != anchors[1]):
            explict_Hs = max_valence[anchor_l_atom_symbol] - anchor_l_atom.GetExplicitValence() - 1
            anchor_l_atom.SetNumExplicitHs(int(explict_Hs))

            explict_Hs = max_valence[anchor_r_atom_symbol] - anchor_r_atom.GetExplicitValence() - 1
            anchor_r_atom.SetNumExplicitHs(int(explict_Hs))
        else:
            explict_Hs = max_valence[anchor_l_atom_symbol] - anchor_l_atom.GetExplicitValence() - 2
            anchor_l_atom.SetNumExplicitHs(int(explict_Hs))

        #add anchors
        if anchor_mode == "thiol":
            sulfur = Chem.MolFromSmiles("S")
            atom = sulfur.GetAtomWithIdx(0)
            atom.SetNumExplicitHs(1)

        elif anchor_mode == "AuS" or anchor_mode == "AuS_just_left":
            sulfur = Chem.MolFromSmiles("S")
            atom = sulfur.GetAtomWithIdx(0)
            atom.SetNumExplicitHs(0)
            gold = Chem.MolFromSmiles("[Au]")
            sulfur = Chem.CombineMols(gold, sulfur)
            sulfur = Chem.RWMol(sulfur)
            sulfur.AddBond(0, 1, Chem.BondType.SINGLE)

        elif anchor_mode == "No":
            pass
        else:
            raise ValueError(f"Unknown anchor mode {anchor_mode}. Use 'thiol' or 'AuS' or 'AuS_just_left'.")

        anchor_l = None
        anchor_r = None
        if anchor_mode == "AuS" or anchor_mode == "thiol":
            # bind to atom with index anchors[0]
            mol = Chem.CombineMols(mol, sulfur)
            mol = Chem.RWMol(mol)
            mol.AddBond(anchors[0], mol.GetNumAtoms() - 1, Chem.BondType.SINGLE)
            # bind to atom with index anchors[1]
            mol = Chem.CombineMols(mol, sulfur)
            mol = Chem.RWMol(mol)
            mol.AddBond(anchors[1], mol.GetNumAtoms() - 1, Chem.BondType.SINGLE)

            anchor_l = mol.GetNumAtoms() - 2
            anchor_r = mol.GetNumAtoms() - 1

            # add hydrogens, but not to the anchors
            mol.UpdatePropertyCache(strict=False)
            mol = Chem.AddHs(mol, onlyOnAtoms=[i for i in range(mol.GetNumAtoms() - 2)], addCoords=True)
            mol = Chem.AddHs(mol, addCoords=True, explicitOnly=True)
            id = Chem.AllChem.EmbedMolecule(mol, randomSeed=0)
            if id == -1:
                raise ValueError("Could not embed molecule")
        elif anchor_mode == "AuS_just_left":
            # bind to atom with index anchors[0]
            mol = Chem.CombineMols(mol, sulfur)
            mol = Chem.RWMol(mol)
            mol.AddBond(anchors[0], mol.GetNumAtoms() - 1, Chem.BondType.SINGLE)

            anchor_l, anchor_r = self.anchors
            anchor_l = mol.GetNumAtoms() - 2

            # add hydrogens, but not to the anchors
            mol.UpdatePropertyCache(strict=False)
            mol = Chem.AddHs(mol, onlyOnAtoms=[i for i in range(mol.GetNumAtoms() - 1)], addCoords=True)
            mol = Chem.AddHs(mol, addCoords=True, explicitOnly=True)
            id = Chem.AllChem.EmbedMolecule(mol, randomSeed=0)
            if id == -1:
                raise ValueError("Could not embed molecule")
        else:
            mol = Chem.AddHs(mol, addCoords=True)
            id = Chem.AllChem.EmbedMolecule(mol, randomSeed=0)
            if id == -1:
                raise ValueError("Could not embed molecule")

            #handling of anchors done


        if save_xyz:
            if save_path is None:
                raise ValueError("No save path given")
            with open(save_path, "w") as f:
                f.write(f"{mol.GetNumAtoms()}\n")
                if anchor_l and anchor_r:
                    f.write(f"{anchor_l}, {anchor_r}\n")
                else:
                    f.write(f"\n")
                for i in range(mol.GetNumAtoms()):
                    pos = mol.GetConformer().GetAtomPosition(i)
                    f.write(f"{mol.GetAtomWithIdx(i).GetSymbol()} {pos.x} {pos.y} {pos.z}\n")

        return mol, [anchor_l, anchor_r]

    def print_grammar_stats(self):
        print("-------Grammar Stats-------")
        for category in self.categorized_grammar:
            print(f"{category=} groups with {category} attachment points")
            for group in category:
                print(f"{group.name=}, {group.attachment_points=}")
                mol = group.mol
                mol = Chem.Mol(mol)
                atoms = mol.GetAtoms()
                for atom in atoms:
                    print(f"{atom.GetSymbol()=}, {atom.GetExplicitValence()=}, {atom.GetNumExplicitHs()=}, {atom.GetNumImplicitHs()=}")
                print("--------------------------")
            print("___________________________")

    def write_bond_constraints(self, path):
        """
        Writes input file for xtb relaxation under constraints. Contains information about connected atoms and sets their
        distance to 1.4.
        :param genome: GGS
        :param path: Calculation path
        """
        #todo: implement this with the corresponding
        mol = self.mol
        adjacency_matrix = Chem.AllChem.GetAdjacencyMatrix(mol)
        with open(path + r'/bond_constraints.inp', 'w') as file:
            file.write('$constrain' + '\n')
            for i in range(adjacency_matrix.shape[0]):
                for j in range(i):
                    if adjacency_matrix[i][j] != 0:
                        #get bond type
                        bond = mol.GetBondBetweenAtoms(i, j)
                        file.write(' distance: ' + str(i + 1) + ', ' + str(j + 1) + ', ' + '1.4' + '\n')
            file.write('$end')

    def draw_mol(self, draw_atom_numbers=False):
        mol = self.mol
        if draw_atom_numbers:
            dos = Draw.MolDrawOptions()
            dos.addAtomIndices = True
            img = Draw.MolToImage(mol, options=dos, size=(500, 500))
        else:
            img = Draw.MolToImage(mol, size=(500, 500))
        img.show()


    def group_mutation(self, allow_same_goup = False, return_new_object = False):
        """
        Mutate the genome by group mutation. The graph and encoding of the genome is not changed.

        :param allow_same_goup: If True, the same group can be selected again. If False, a different group is selected if possible.
        :param return_new_object: If True, a new genome object is returned. If False, the encoding and graph are returned.
        :return: (new_encoding, new_graph) or new object
        :raises ValueError: If no allowed group for fragment mutation is found
        """

        #(f"starting with encoding {self.encoding}")
        #select random node in graph
        nodes = list(self._selfies_graph.nodes)
        node = self.rng.choice(nodes)
        neighbors = list(self._selfies_graph.neighbors(node))
        #edges to neighbors
        edges = [self._selfies_graph.edges[node, neighbor] for neighbor in neighbors]
        predecessor = [n for n in self._selfies_graph.predecessors(node)]
        if len(predecessor) > 1:
            raise ValueError("More than one predecessor")


        #get the group
        group = node
        #get occupied attachment points
        occupied_attachment_points = group.occupied_attachment_points
        #get reserved attachment point
        n_reserved_attachment_point = 0
        if group.reserved_attachment_point != -1:
            n_reserved_attachment_point = 1


        #select new group. Category must allow number of occupied attachment points. Keep reserved attachment point in mind
        category = lambda n: n - 1 if n < 3 else 2
        category = category(len(occupied_attachment_points)+n_reserved_attachment_point)
        allowed_categories = []
        #todo: This can be simplified. For sake of clarity, it is not simplified
        if category == 0:
            allowed_categories.append(0)
            allowed_categories.append(1)
            allowed_categories.append(2)
        elif category == 1:
            allowed_categories.append(1)
            allowed_categories.append(2)
        else:
            allowed_categories.append(2)

        #select category
        category = self.rng.choice(allowed_categories)

        #select group: We have to select a group with more or equal attachment points than the current group
        #selecting the right category is not enough
        #We do not want to select the same group
        allowed_groups = self.categorized_grammar[category]
        if allow_same_goup:
            allowed_groups = [item for item in allowed_groups if len(item.attachment_points) >= len(group.occupied_attachment_points)]
        else:
            allowed_groups = [item for item in allowed_groups if len(item.attachment_points) >= len(group.occupied_attachment_points) and item.name != group.name]

        if len(allowed_groups) == 0:
            raise ValueError(f"No allowed group for fragment mutation found in {self.encoding}")

        new_group = self.rng.choice(allowed_groups)
        new_group = Group(new_group.name, new_group.canonsmiles)

        #print(f"Mutate fragment: {group.name} -> {new_group.name}")

        #copy graph
        new_graph = self._selfies_graph.copy()

        #replace group in graph. #TODO: mabye we should not remove the node, but just replace it
        new_graph.remove_node(node)
        new_graph.add_node(new_group, group=new_group)

        #handle edge to predecessor
        shift = new_group.next_avail_attachment(occupied_attachment_points[0])
        new_group.occupy_attachment_point(shift)
        #take care of predecessor
        if len(predecessor) > 0:
            predecessor = predecessor[0]
            #print("edges", self._selfies_graph.edges)
            edge = self._selfies_graph.edges[predecessor, group]
            edge_start_attachment_point = edge['start_attachment_point']
            edge_end_attachment_point = edge['end_attachment_point']
            new_graph.add_edge(predecessor, new_group, start_attachment_point=edge_start_attachment_point, end_attachment_point=edge_end_attachment_point)


        #handle reserved attachment point
        if group.reserved_attachment_point != -1:
            reserved_shift = group.reserved_attachment_point
            reserved_shift = new_group.next_avail_attachment(reserved_shift)
            new_group.reserve_attachment_point(reserved_shift)


        #handle edges to neighbors
        for i, neighbor in enumerate(neighbors):
            edge = edges[i]
            #print("taking care of edge", edge)
            edge_start_attachment_point = edge['start_attachment_point']
            edge_end_attachment_point = edge['end_attachment_point']
            shift = new_group.next_avail_attachment(edge_start_attachment_point)
            new_group.occupy_attachment_point(shift)
            new_graph.add_edge(new_group, neighbor, start_attachment_point=edge_start_attachment_point, end_attachment_point=edge_end_attachment_point)

        self.print_graph(selfies_graph=new_graph)
        new_encoding = self.graph_to_encoding(new_graph)
        #print(f"new encoding {new_encoding}")

        if return_new_object:
            return GGS(new_encoding, self.grammar)

        return new_encoding, new_graph

    def bond_mutation(self, return_new_object=False):
        """
        Mutate the genome by bond mutation. The graph and encoding of the genome is not changed. Only bonds between
        groups are changed. Anchor positions are not changed.
        :param return_new_object: If True, a new genome object is returned. If False, the encoding and graph are returned.
        :return: (new_encoding, new_graph) or new object
        :raises ValueError: If no allowed node for bond mutation is found
        """

        #deep copy of graph
        encoding = self.encoding
        #do i need to construct a new object here?
        ggs_copy = GGS(encoding, self.grammar, self.rng)
        new_graph = ggs_copy._selfies_graph

        # find suitable node for bond mutation
        nodes = list(new_graph)
        nodes_n_free_attach_points = [len(node.attachment_points)-len(node.occupied_attachment_points) for node in nodes]
        depth = lambda node: len(list(nx.descendants(new_graph, node)))

        allowed_nodes = [node for i, node in enumerate(nodes) if nodes_n_free_attach_points[i] > 0 and depth(node) > 0]
        if len(allowed_nodes) == 0:
            raise ValueError(f"No allowed node for bond mutation found in {self.encoding}")

        node = self.rng.choice(allowed_nodes)
        #print(f"Chose {node} with {node.occupied_attachment_points}")
        neighbors = list(new_graph.neighbors(node))
        edges = [(new_graph.edges[node, neighbor], neighbor) for neighbor in neighbors]

        #choose edge
        edge = self.rng.choice(edges)
        #print(f"Taking {edge=}")
        edge_start_attachment_point = edge[0]['start_attachment_point']
        edge_end_attachment_point = edge[0]['end_attachment_point']

        node.free_attachment_point(edge_start_attachment_point)
        shift = self.rng.integers(0, len(node.attachment_points))
        edge_start_attachment_point = node.next_avail_attachment(shift)
        node.occupy_attachment_point(edge_start_attachment_point)

        #print(f"Making it to {edge_start_attachment_point=}, {edge_end_attachment_point=}")

        #update graph
        new_graph.remove_edge(node, edge[1])
        new_graph.add_edge(node, edge[1], start_attachment_point=edge_start_attachment_point, end_attachment_point=edge_end_attachment_point)

        new_encoding = self.graph_to_encoding(new_graph)

        if return_new_object:
            return GGS(new_encoding, self.grammar, rng = self.rng)

        return new_encoding, new_graph


    def anchor_pos_mutation(self, return_new_object=False):
        """
        Mutate the genome by anchor position mutation (the group/fragment of the anchor position is not changed). The graph and encoding of the genome is not changed.
        :param return_new_object: If True, a new genome object is returned. If False, the encoding and graph are returned.
        :return: (new_encoding, new_graph) or new object
        :raises ValueError: If no allowed node for bond mutation is found
        """

        graph_copy = self._selfies_graph.copy()


        #left anchor pos in corresponding node : start_attachment_point
        #right anchor pos in corresponding node : reserved_attachment_point
        rand_float = self.rng.random()
        #mutate left anchor position
        if rand_float < 0.5:

            #find start node -> left anchor node
            start_node = [node for node in graph_copy.nodes if graph_copy.in_degree(node) == 0]
            assert len(start_node) == 1
            start_node = start_node[0]

            free_attachment_points = start_node.get_free_attachment_points()
            if len(free_attachment_points) == 0:
                raise ValueError(f"No free attachment points available in {start_node}")

            #choose random attachment point
            attachment_point = self.rng.choice(free_attachment_points)
            #free old attachment point
            start_node.reset_attachment_point_start()
            start_node.attachment_point_start = attachment_point

        #mutate right anchor position
        else:
            #find node with reserved attachment point -> right anchor node
            end_node = [node for node in graph_copy.nodes if node.reserved_attachment_point != -1]
            assert len(end_node) == 1
            end_node = end_node[0]

            #free attachment points
            free_attachment_points = end_node.get_free_attachment_points()
            if len(free_attachment_points) == 0:
                raise ValueError(f"No free attachment points available in {end_node}")

            #choose random attachment point
            attachment_point = self.rng.choice(free_attachment_points)
            #free old reserved attachment point
            end_node.reserved_attachment_point = attachment_point


        new_encoding = self.graph_to_encoding(graph_copy)

        if return_new_object:
            return GGS(new_encoding, self.grammar, rng = self.rng)
        return new_encoding, graph_copy

    def anchor_group_mutation(self, return_new_object=False):

        graph_copy = self._selfies_graph.copy()

        #find start and end node in current graph
        start_node = [node for node in graph_copy.nodes if graph_copy.in_degree(node) == 0]
        assert len(start_node) == 1
        start_node = start_node[0]

        end_node = [node for node in graph_copy.nodes if node.reserved_attachment_point != -1]
        assert len(end_node) == 1
        end_node = end_node[0]

        #---------------- change end node ----------------
        ##find predecessor of end node
        #predecessors = list(graph_copy.predecessors(end_node))
        #assert len(predecessors) == 1, f"More than one predecessor found for end node {end_node} in {self.encoding}"
        #predecessor = predecessors[0]
        neighbors = list(graph_copy.neighbors(end_node))
        allowed_new_end_nodes = [node for node in neighbors if len(list(graph_copy.neighbors(node))) == 0
                                 and node.reserved_attachment_point == -1 and node != end_node
                                 and node.next_avail_attachment(0) != -1]
        new_end_node_possible = (len(allowed_new_end_nodes) != 0)
        if new_end_node_possible:
            new_end_node = self.rng.choice(allowed_new_end_nodes)
            shift = new_end_node.next_avail_attachment(0)
            new_end_node.reserve_attachment_point(shift)
            end_node.reserved_attachment_point = -1

            new_encoding = self.graph_to_encoding(graph_copy)
            if return_new_object:
                return GGS(new_encoding, self.grammar, rng = self.rng)
            return new_encoding, graph_copy

        # ---------------- change start node ----------------
        successors = list(graph_copy.successors(start_node))
        allowed_new_start_nodes = [node for node in successors if len(list(graph_copy.neighbors(node))) == 0 and node != start_node and node.next_avail_attachment(0) != -1]
        new_start_node_possible = (len(allowed_new_start_nodes) != 0)
        if new_start_node_possible:
            new_start_node = self.rng.choice(allowed_new_start_nodes)
            edge = graph_copy.edges[start_node, new_start_node]
            edge_start_attachment_point = edge['start_attachment_point']
            edge_end_attachment_point = edge['end_attachment_point']
            graph_copy.remove_edge(start_node, new_start_node)
            graph_copy.add_edge(new_start_node, start_node, start_attachment_point=edge_end_attachment_point, end_attachment_point=edge_start_attachment_point)
            new_start_node.reset_attachment_point_start()
            shift = self.rng.choice(new_start_node.attachment_points)
            shift = new_start_node.next_avail_attachment(shift)
            new_start_node.occupy_attachment_point(shift)

            new_encoding = self.graph_to_encoding(graph_copy)
            if return_new_object:
                return GGS(new_encoding, self.grammar, rng = self.rng)
            return new_encoding, graph_copy

        raise ValueError("No allowed node for anchor group mutation found in {self.encoding}. ")



    def insert_group_mutation(self, return_new_object=False):
        """
        Inserts a new group in the middle of the branch
        """
        new_graph = self._selfies_graph.copy()
        nodes = list(self._selfies_graph.nodes)
        allowed_nodes = [node for node in nodes if len(list(new_graph.predecessors(node))) > 0 and (len(list(new_graph.neighbors(node))) > 0 or node.reserved_attachment_point != -1)]
        if (len(allowed_nodes) == 0):
            raise ValueError("No allowed node found in insert group mutation")
        node = self.rng.choice(allowed_nodes)

        predecessors = list(new_graph.predecessors(node))
        if len(predecessors) > 1:
            raise ValueError("More than one predecessor found in insert group mutation")
        predecessor = predecessors[0]

        allowed_categories = [1, 2]
        category = self.rng.choice(allowed_categories)
        allowed_groups = self.categorized_grammar[category]
        new_group = self.rng.choice(allowed_groups)
        new_group = Group(new_group.name, new_group.canonsmiles)

        old_edge = new_graph.edges[predecessor, node]
        old_edge_start_attachment_point = old_edge['start_attachment_point']
        old_edge_end_attachment_point = old_edge['end_attachment_point']

        # handle new_group
        new_graph.add_node(new_group, group=new_group)
        shift_in = old_edge_end_attachment_point
        shift_in = new_group.next_avail_attachment(shift_in)
        new_group.occupy_attachment_point(shift_in)
        shift_out = self.rng.choice(new_group.attachment_points)
        shift_out = new_group.next_avail_attachment(shift_out)
        new_group.occupy_attachment_point(shift_out)


        new_graph.remove_edge(predecessor, node)
        new_graph.add_edge(predecessor, new_group, start_attachment_point = old_edge_start_attachment_point, end_attachment_point = shift_in)
        new_graph.add_edge(new_group, node, start_attachment_point=shift_out, end_attachment_point=old_edge_end_attachment_point)

        new_encoding = self.graph_to_encoding(new_graph)
        if return_new_object:
            return GGS(new_encoding, self.grammar, rng = self.rng)
        return new_encoding, new_graph

        new_encoding = self.graph_to_encoding(new_graph)

        if return_new_object:
            return GGS(new_encoding, self.grammar, rng = self.rng)
        return new_encoding, new_graph



    def insert_start_end_group_mutation(self, return_new_object=False):
        """
        Inserts a new group at the beginning or end -> might be improved by inserting at a random position.
        """
        graph_copy = self._selfies_graph.copy()

        #choose new group
        allowed_categories = [1, 2]
        category = self.rng.choice(allowed_categories)
        allowed_groups = self.categorized_grammar[category]
        new_group = self.rng.choice(allowed_groups)
        new_group = Group(new_group.name, new_group.canonsmiles)
        new_graph = self._selfies_graph.copy()

        #handle new_group
        new_graph.add_node(new_group, group=new_group)
        shift_in = self.rng.choice(new_group.attachment_points)
        shift_in = new_group.next_avail_attachment(shift_in)
        new_group.occupy_attachment_point(shift_in)
        shift_out = self.rng.choice(new_group.attachment_points)
        shift_out = new_group.next_avail_attachment(shift_out)
        new_group.reserve_attachment_point(shift_out)

        rand_float = self.rng.random()
        #mutate left anchor position -> beginning of the graph
        if rand_float < 0.5:

            #find start node -> left anchor node
            start_node = [node for node in graph_copy.nodes if graph_copy.in_degree(node) == 0]
            assert len(start_node) == 1
            start_node = start_node[0]

            edge_start_attachment_point = new_group.reserved_attachment_point
            edge_end_attachment_point = start_node.attachment_point_start

            new_graph.add_edge(new_group, start_node, start_attachment_point=edge_start_attachment_point, end_attachment_point=edge_end_attachment_point)
            new_group.occupy_attachment_point(new_group.reserved_attachment_point)

        else:
            end_node = [node for node in graph_copy.nodes if node.reserved_attachment_point != -1]
            assert len(end_node) == 1
            end_node = end_node[0]

            edge_start_attachment_point = end_node.reserved_attachment_point
            edge_end_attachment_point = new_group.attachment_point_start

            new_graph.add_edge(end_node, new_group, start_attachment_point=edge_start_attachment_point, end_attachment_point=edge_end_attachment_point)
            end_node.occupy_attachment_point(end_node.reserved_attachment_point)



        new_encoding = self.graph_to_encoding(new_graph)

        if return_new_object:
            return GGS(new_encoding, self.grammar, rng = self.rng)
        return new_encoding, new_graph

    def truncate_mutation(self, return_new_object=False):
        """removes a group in the graph"""

        new_graph = self._selfies_graph.copy()
        nodes = list(self._selfies_graph.nodes)
        allowed_nodes = [node for node in nodes if node.reserved_attachment_point == -1]
        if(len(allowed_nodes) == 0):
            raise ValueError("No allowed node found in truncate mutation")
        node = self.rng.choice(allowed_nodes)

        predecessors = list(new_graph.predecessors(node))

        #truncate branch -> node with no successors and no reserved attachment point
        if len(list(new_graph.successors(node))) == 0 and len(predecessors) > 0 and node.reserved_attachment_point == -1:

            assert len(predecessors) == 1
            predecessor = predecessors[0]

            #get edge to predecessor
            edge = new_graph.edges[predecessor, node]
            edge_start_attachment_point = edge['start_attachment_point']
            predecessor.free_attachment_point(edge_start_attachment_point)

            new_graph.remove_node(node)

        #truncate middle node -> node with successors and predecessor and no reserved attachment point
        elif len(list(new_graph.successors(node))) > 0 and len(predecessors) > 0 and node.reserved_attachment_point == -1:

            assert len(predecessors) == 1
            predecessor = predecessors[0]

            edge = new_graph.edges[predecessor, node]
            edge_start_attachment_point = edge['start_attachment_point']

            # choose a neighbor where a node with reserved attachment point is reachable. Since we have maximum pop depth of
            # 1 this can be judged if the neighbor has a reserved attachment point or len(neighbors) > 0
            neighbors = list(new_graph.neighbors(node))
            allowed_successor = [neighbor for neighbor in neighbors if len(list(new_graph.neighbors(neighbor))) > 0 or neighbor.reserved_attachment_point != -1]
            successor = allowed_successor[0]
            edge_end_attachment_point = successor.attachment_point_start

            #remove nodes without any connection
            for neighbor in neighbors:
                if neighbor != successor:
                    new_graph.remove_node(neighbor)

            new_graph.remove_node(node)
            new_graph.add_edge(predecessor, successor, start_attachment_point=edge_start_attachment_point, end_attachment_point=edge_end_attachment_point)
        #remove start node -> works only if molecule has more than one node
        elif len(list(new_graph.successors(node))) > 0 and node.reserved_attachment_point == -1:
            neighbors = list(new_graph.neighbors(node))
            allowed_successor = [neighbor for neighbor in neighbors if len(list(new_graph.neighbors(neighbor))) > 0 or neighbor.reserved_attachment_point != -1]
            successor = allowed_successor[0]
            for neighbor in neighbors:
                if neighbor != successor:
                    new_graph.remove_node(neighbor)
            new_graph.remove_node(node)
        #remove end node -> works only if molecule has more than one node
        elif len(predecessors) > 0 and node.reserved_attachment_point == -1:
            assert len(predecessors) == 1
            predecessor = predecessors[0]
            # get edge to predecessor
            edge = new_graph.edges[predecessor, node]
            edge_start_attachment_point = edge['start_attachment_point']
            predecessor.free_attachment_point(edge_start_attachment_point)
            predecessor.reserve_attachment_point(edge_start_attachment_point)

            neighbors = list(new_graph.neighbors(node))
            for neigbor in neighbors:
                new_graph.remove_node(neigbor)
            new_graph.remove_node(node)


        new_encoding = self.graph_to_encoding(new_graph)

        if return_new_object:
            return GGS(new_encoding, self.grammar, rng = self.rng)
        return new_encoding, new_graph


    def insert_branch_mutation(self, return_new_object=False):
        """
        inserts branch at any position in the graph.
        """

        #get groups which have a successor and free attachment points
        nodes = list(self._selfies_graph.nodes)
        allowed_nodes = [node for node in nodes if len(list(self._selfies_graph.successors(node))) > 0 and len(node.get_free_attachment_points()) > 0]
        if len(allowed_nodes) == 0:
            raise ValueError("No allowed node for insert branch mutation found in {self.encoding}")
        node = self.rng.choice(allowed_nodes)
        free_attachment_points = node.get_free_attachment_points()
        attachment_point = self.rng.choice(free_attachment_points)
        node.reserve_attachment_point(attachment_point)

        #choose new group
        allowed_categories = [0, 1, 2]
        category = self.rng.choice(allowed_categories)
        allowed_groups = self.categorized_grammar[category]
        new_group = self.rng.choice(allowed_groups)
        new_group = Group(new_group.name, new_group.canonsmiles)
        shift_in = self.rng.choice(new_group.attachment_points)
        shift_in = new_group.next_avail_attachment(shift_in)
        new_group.occupy_attachment_point(shift_in)

        new_graph = self._selfies_graph.copy()
        new_graph.add_node(new_group, group=new_group)

        edge_start_attachment_pint = node.reserved_attachment_point
        edge_end_attachment_pint = new_group.attachment_point_start

        new_graph.add_edge(node, new_group, start_attachment_point=edge_start_attachment_pint, end_attachment_point=edge_end_attachment_pint)
        node.occupy_attachment_point(node.reserved_attachment_point)

        new_encoding = self.graph_to_encoding(new_graph)

        if return_new_object:
            return GGS(new_encoding, self.grammar, rng = self.rng)
        return new_encoding, new_graph



    def single_point_crossover(self, Genome_1, Genome_2, return_new_object = False, draw_steps=False):
        """
        Perform a single point crossover between two genomes. The genomes are not changed.
        :param Genome_1:
        :param Genome_2:
        :param draw_steps:
        :raise ValueError: If no allowed neighbor for crossover is found
        :return: (new_encoding, new_graph), (new_encoding, new_graph) or (new_genome, new_genome)
        """

        encoding = Genome_1.encoding
        Genome_1 = GGS(encoding, self.grammar, rng = self.rng)

        encoding = Genome_2.encoding
        Genome_2 = GGS(encoding, self.grammar, rng = self.rng)

        nodes_1 = list(Genome_1._selfies_graph.nodes)
        nodes_2 = list(Genome_2._selfies_graph.nodes)
        allowed_nodes_1 = [node for node in nodes_1 if len(list(Genome_1._selfies_graph.neighbors(node))) > 0 and node.reserved_attachment_point == -1]
        allowed_nodes_2 = [node for node in nodes_2 if len(list(Genome_2._selfies_graph.neighbors(node))) > 0 and node.reserved_attachment_point == -1]

        if len(allowed_nodes_1) == 0 or len(allowed_nodes_2) == 0:
            raise ValueError(f"No allowed node for crossover found in {Genome_1.encoding} or {Genome_2.encoding}")


        n_nodes_total = len(nodes_1) + len(nodes_2)
        n_edges_total = len(Genome_1._selfies_graph.edges) + len(Genome_2._selfies_graph.edges)

        '''
        if len(nodes_1) > len(nodes_2):
            longer, shorter = nodes_1, nodes_2
        else:
            longer, shorter = nodes_2, nodes_1

        shorter_length = len(shorter)
        longer_length = len(longer)'''

        #choose cut in shorter genome
        cut = self.rng.integers(0, len(allowed_nodes_1)-1)

        graph_1 = Genome_1._selfies_graph
        #The sorting was an attempt to preserve the length of the genome. This does not work
        #node_1 = nodes_1[cut]
        node_1 = self.rng.choice(allowed_nodes_1)
        neighbors_1 = list(Genome_1._selfies_graph.neighbors(node_1))
        #choose a neighbor as successor where a node with reserved attachment point is reachable. Since we have maximum pop depth of
        #1 this can be judged if the neighbor has a reserved attachment point or len(neighbors) > 0
        allowed_successor_1 = [neighbor for neighbor in neighbors_1 if len(list(graph_1.neighbors(neighbor))) > 0 or neighbor.reserved_attachment_point != -1]
        if len(allowed_successor_1) == 0:
            #print(f"No allowed neighbor for crossover in {Genome_1} found")
            raise ValueError(f"No allowed neighbor for crossover in {Genome_1} found")
        chosen_successor_1 = self.rng.choice(allowed_successor_1)
        edge = graph_1.edges[node_1, chosen_successor_1]
        edge_start_attachment_point_1 = edge['start_attachment_point']
        edge_end_attachment_point_1 = edge['end_attachment_point']

        graph_2 = Genome_2._selfies_graph
        #cut = self.rng.randint(0, len(allowed_nodes_2) - 1)
        #node_2 = nodes_2[cut]
        node_2 = self.rng.choice(allowed_nodes_2)
        neighbors_2 = list(Genome_2._selfies_graph.neighbors(node_2))
        # choose a neighbor where a node with reserved attachment point is reachable. Since we have maximum pop depth of
        # 1 this can be judged if the neighbor has a reserved attachment point or len(neighbors) > 0
        allowed_successor_2 = [neighbor for neighbor in neighbors_2 if len(list(graph_2.neighbors(neighbor))) > 0 or neighbor.reserved_attachment_point != -1]
        if len(allowed_successor_2) == 0:
            #print(f"No allowed neighbor for crossover in {Genome_2} found")
            raise ValueError(f"No allowed neighbor for crossover in {Genome_2} found")
        chosen_successor_2 = self.rng.choice(allowed_successor_2)
        edge = graph_2.edges[node_2, chosen_successor_2]
        edge_start_attachment_point_2 = edge['start_attachment_point']
        edge_end_attachment_point_2 = edge['end_attachment_point']

        descendants_1 = list(nx.descendants(graph_1, chosen_successor_1))
        descendants_1.append(chosen_successor_1)
        descendants_2 = list(nx.descendants(graph_2, chosen_successor_2))
        descendants_2.append(chosen_successor_2)
        desc_sub_graph_1 = graph_1.subgraph(descendants_1).copy()
        desc_sub_graph_2 = graph_2.subgraph(descendants_2).copy()

        if draw_steps:
            # draw initial graphs
            nx.draw(graph_1, with_labels=True, node_color='blue')
            plt.show()

            nx.draw(graph_2, with_labels=True, node_color='green')
            plt.show()


        graph_1.remove_edge(node_1, chosen_successor_1)
        graph_2.remove_edge(node_2, chosen_successor_2)



        if draw_steps:
            #draw split graph
            nx.draw(desc_sub_graph_2, with_labels=True, node_color='red')
            plt.show()

            nx.draw(desc_sub_graph_1, with_labels=True, node_color='orange')
            plt.show()


        #remove all nodes following the chosen neighbors
        for node in descendants_1:
            graph_1.remove_node(node)

        for node in descendants_2:
            graph_2.remove_node(node)


        beginning_desc_sub_graph_2 = [node for node in desc_sub_graph_2.nodes if desc_sub_graph_2.in_degree(node) == 0]
        assert len(beginning_desc_sub_graph_2) == 1
        edges = [(edge[0], edge[1], desc_sub_graph_2.edges[edge[0], edge[1]]) for edge in desc_sub_graph_2.edges]
        graph_1.update(edges=edges, nodes=desc_sub_graph_2.nodes)
        graph_1.add_edge(node_1, beginning_desc_sub_graph_2[0], start_attachment_point=edge_start_attachment_point_1, end_attachment_point=edge_end_attachment_point_2, added_edge="JA")

        beginning_desc_sub_graph_1 = [node for node in desc_sub_graph_1.nodes if desc_sub_graph_1.in_degree(node) == 0]
        assert len(beginning_desc_sub_graph_1) == 1
        edges = [(edge[0], edge[1], desc_sub_graph_1.edges[edge[0], edge[1]]) for edge in desc_sub_graph_1.edges]
        graph_2.update(edges=edges, nodes=desc_sub_graph_1.nodes)
        graph_2.add_edge(node_2, beginning_desc_sub_graph_1[0], start_attachment_point=edge_start_attachment_point_2, end_attachment_point=edge_end_attachment_point_1, added_edge="JA")

        if draw_steps:
            #draw new graphs
            nx.draw(graph_1, with_labels=True, node_color='purple')
            plt.show()

            nx.draw(graph_2, with_labels=True, node_color='brown')
            plt.show()

        len_graph_1 = len(list(graph_1))
        len_graph_2 = len(list(graph_2))
        assert n_nodes_total == len_graph_1 + len_graph_2
        assert n_edges_total == len(graph_1.edges) + len(graph_2.edges)

        new_encoding_1 = Genome_1.graph_to_encoding(graph_1)
        new_encoding_2 = Genome_2.graph_to_encoding(graph_2)

        if return_new_object:
            return GGS(new_encoding_1, self.grammar, rng = self.rng), GGS(new_encoding_2, self.grammar, rng = self.rng)

        return (new_encoding_1, graph_1), (new_encoding_2, graph_2)





def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    import rdkit.rdBase as rkrb
    import rdkit.RDLogger as rkl
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog('rdApp.error')

if __name__ == '__main__':

    disable_rdkit_logging()

    grammar_path = "./data/GS_complex_grammar.txt"
    gs1 = "[:1frag_31][Branch][:0frag_33][pop][Branch][:4frag_6][Ring1][:0frag_58][Ring1][:4frag_6][Ring1][:3frag_70][=Branch]"
    gs1 = "[:4frag_0][Branch][:0frag_33][pop][Ring1][:0frag_64][pop][Ring1][:5frag_12][Ring1]"
    #gs1 = "[:1frag_5][Branch][:5frag_0][Branch][:6frag_12][N][:0frag_33][pop][Branch][:1frag_58][Ring1][:2frag_17][Ring1]"
    gs1 = GGS(gs1, grammar_path=grammar_path)
    gs1.draw_mol()
    gs1 = gs1.insert_group_mutation(return_new_object=True)
    gs1.draw_mol()



    """
    #gs2 = GGS(gs2, grammar_path=grammar_path)
    #gs = GGS(grammar_path=grammar_path)
    #gs.single_point_crossover(gs1, gs2, return_new_object=True)


    invalid = 0
    for i in range(0,100):
        print(i)
        gs1 = GGS(grammar_path=grammar_path)
        gs1.create_random_genome(10,5)
        gs2 = GGS(grammar_path=grammar_path)
        gs2.create_random_genome(4,6)
        gs = GGS(grammar_path=grammar_path)
        print(gs1.encoding)
        print(gs2.encoding)
        try:
            gs, gs2 = gs.single_point_crossover(gs1, gs2, return_new_object=True, draw_steps=False)
        except ValueError as ex:
            invalid += 1
            print("Error in crossover", ex)
        print("--------------")

    print(f"Invalid crossovers: {invalid}")
    """




