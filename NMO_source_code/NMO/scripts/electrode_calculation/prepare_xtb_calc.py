import sys

import numpy as np
import rdkit.Chem as Chem
from .utils import read_xyz_file, write_xyz_file, align_molecule
def remove_sulfur_hydrogen(coord_xyz):
    """
    Remove the hydrogen atoms which saturate the sulfur atoms in the molecule.
    :param coord_xyz:
    :return:
    """
    #find indices where coord_xyz[0,i] is "S"
    sulfur_indices = [i for i, x in enumerate(coord_xyz[0]) if x == "S"]
    assert len(sulfur_indices) == 2, "Molecule must have exactly 2 sulfur atoms"
    hydogen_indices = [i for i, x in enumerate(coord_xyz[0]) if x == "H"]
    #calculate distance between sulfur atom 0 and all hydrogen atoms
    distances = []
    for index in hydogen_indices:
        distances.append(np.linalg.norm(coord_xyz[1:4, sulfur_indices[0]] - coord_xyz[1:4, index]))
    min_distance_index_1 = np.argmin(distances)
    #calculate distance between sulfur atom 1 and all hydrogen atoms
    distances = []
    for index in hydogen_indices:
        distances.append(np.linalg.norm(coord_xyz[1:4, sulfur_indices[1]] - coord_xyz[1:4, index]))
    min_distance_index_2 = np.argmin(distances)
    print(f"min_distance_index_1: {min_distance_index_1}")
    print(f"min_distance_index_2: {min_distance_index_2}")
    assert min_distance_index_1 != min_distance_index_2, "Both hydrogen atoms have the same index"
    if min_distance_index_2 < min_distance_index_1:
        #swap sulfur indices
        min_distance_index_2, min_distance_index_1 = min_distance_index_1, min_distance_index_2
    #remove hydrogen atoms from coord_xyz
    print([hydogen_indices[min_distance_index_1], hydogen_indices[min_distance_index_2]])
    coord_xyz = np.delete(coord_xyz, [hydogen_indices[min_distance_index_1], hydogen_indices[min_distance_index_2]], axis=1)
    return coord_xyz


if __name__ == "__main__":
    g_surf_left_path = ""
    g_surf_right_path = ""

    coord_xyz_in = "./utils/xtbopt.xyz"
    #coord_xyz_out = sys.argv[2]

    #read inputfile using rdkit
    coord_xyz = read_xyz_file(coord_xyz_in)

    #remove hydrogen atoms at the sulfur atoms
    coord_xyz = remove_sulfur_hydrogen(coord_xyz)

    #align molecule
    sulfur_indices = [i for i, x in enumerate(coord_xyz[0]) if x == "S"]
    assert len(sulfur_indices) == 2, "Molecule must have exactly 2 sulfur atoms"
    coord_xyz = align_molecule(coord_xyz, [0,0,1], sulfur_indices)
    coord_s_left = coord_xyz[1:4, sulfur_indices[0]]
    coord_s_right = coord_xyz[1:4, sulfur_indices[1]]
    if coord_s_left[2] > coord_s_right[2]:
        coord_s_left, coord_s_right = coord_s_right, coord_s_left

    #load left tip
    coord_left = read_xyz_file("./example/Au_111_3x4x6_left_hollow.xyz")
    indices_to_keep = [i for i, x in enumerate(coord_left[0]) if x == "Au"]
    print(f"indices_to_keep: {indices_to_keep}")
    coord_left = coord_left[:, indices_to_keep]
    coord_left[1:4, :] += coord_s_left[:, np.newaxis]

    #load right tip
    coord_right = read_xyz_file("./example/Au_111_3x4x6_right_hollow.xyz")
    indices_to_keep = [i for i, x in enumerate(coord_right[0]) if x == "Au"]
    print(f"indices_to_keep: {indices_to_keep}")
    coord_right = coord_right[:, indices_to_keep]
    coord_right[1:4, :] += coord_s_right[:, np.newaxis]



    #concatenate left and right tip along axis 1
    coord_xyz = np.concatenate((coord_left, coord_xyz, coord_right), axis=1)


    write_xyz_file("./utils/coord.xyz", coord_xyz)
