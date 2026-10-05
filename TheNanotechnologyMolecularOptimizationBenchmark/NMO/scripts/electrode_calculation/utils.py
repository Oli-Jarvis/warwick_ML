import numpy as np
from scipy.interpolate import PchipInterpolator
import re
import torch
import torch.nn.functional as F
import tad_mctc as mctc
from rdkit import Chem
import h5py
import io
import math
import random


ANG2BOHR = 1.88973

ATOM_DICT_SYM = {'h': [1, 'Hydrogen', 1.008],
        'he': [2, 'Helium', 4.0026022],
        'li': [3, 'Lithium', 6.94],
        'be': [4, 'Beryllium', 9.01218315],
        'b': [5, 'Boron', 10.81],
        'c': [6, 'Carbon', 12.011],
        'n': [7, 'Nitrogen', 14.007],
        'o': [8, 'Oxygen', 15.999],
        'f': [9, 'Fluorine', 18.9984031635],
        'ne': [10, 'Neon', 20.17976],
        'na': [11, 'Sodium', 22.989769282],
        'mg': [12, 'Magnesium', 24.305],
        'al': [13, 'Aluminium', 26.98153843],
        'si': [14, 'Silicon', 28.085],
        'p': [15, 'Phosphorus', 30.9737619985],
        's': [16, 'Sulfur', 32.06],
        'cl': [17, 'Chlorine', 35.45],
        'ar': [18, 'Argon', 39.95],
        'k': [19, 'Potassium', 39.09831],
        'ca': [20, 'Calcium', 40.0784],
        'sc': [21, 'Scandium', 44.9559074],
        'ti': [22, 'Titanium', 47.8671],
        'v': [23, 'Vanadium', 50.94151],
        'cr': [24, 'Chromium', 51.99616],
        'mn': [25, 'Manganese', 54.9380432],
        'fe': [26, 'Iron', 55.8452],
        'co': [27, 'Cobalt', 58.9331943],
        'ni': [28, 'Nickel', 58.69344],
        'cu': [29, 'Copper', 63.5463],
        'zn': [30, 'Zinc', 65.382],
        'ga': [31, 'Gallium', 69.7231],
        'ge': [32, 'Germanium', 72.6308],
        'as': [33, 'Arsenic', 74.9215956],
        'se': [34, 'Selenium', 78.9718],
        'br': [35, 'Bromine', 79.904],
        'kr': [36, 'Krypton', 83.7982],
        'rb': [37, 'Rubidium', 85.46783],
        'sr': [38, 'Strontium', 87.621],
        'y': [39, 'Yttrium', 88.9058382],
        'zr': [40, 'Zirconium', 91.2242],
        'nb': [41, 'Niobium', 92.906371],
        'mo': [42, 'Molybdenum', 95.951],
        'tc': [43, 'Technetium', 97.0],
        'ru': [44, 'Ruthenium', 101.072],
        'rh': [45, 'Rhodium', 102.905492],
        'pd': [46, 'Palladium', 106.421],
        'ag': [47, 'Silver', 107.86822],
        'cd': [48, 'Cadmium', 112.4144],
        'in': [49, 'Indium', 114.8181],
        'sn': [50, 'Tin', 118.7107],
        'sb': [51, 'Antimony', 121.7601],
        'te': [52, 'Tellurium', 127.603],
        'i': [53, 'Iodine', 126.904473],
        'xe': [54, 'Xenon', 131.2936],
        'cs': [55, 'Caesium', 132.905451966],
        'ba': [56, 'Barium', 137.3277],
        'la': [57, 'Lanthanum', 138.905477],
        'ce': [58, 'Cerium', 140.1161],
        'pr': [59, 'Praseodymium', 140.907661],
        'nd': [60, 'Neodymium', 144.2423],
        'pm': [61, 'Promethium', 145.0],
        'sm': [62, 'Samarium', 150.362],
        'eu': [63, 'Europium', 151.9641],
        'gd': [64, 'Gadolinium', 157.253],
        'tb': [65, 'Terbium', 158.9253547],
        'dy': [66, 'Dysprosium', 162.5001],
        'ho': [67, 'Holmium', 164.9303295],
        'er': [68, 'Erbium', 167.2593],
        'tm': [69, 'Thulium', 168.9342195],
        'yb': [70, 'Ytterbium', 173.0451],
        'lu': [71, 'Lutetium', 174.96681],
        'hf': [72, 'Hafnium', 178.4866],
        'ta': [73, 'Tantalum', 180.947882],
        'w': [74, 'Tungsten', 183.841],
        're': [75, 'Rhenium', 186.2071],
        'os': [76, 'Osmium', 190.233],
        'ir': [77, 'Iridium', 192.2172],
        'pt': [78, 'Platinum', 195.0849],
        'au': [79, 'Gold', 196.9665704],
        'hg': [80, 'Mercury', 200.5923],
        'tl': [81, 'Thallium', 204.38],
        'pb': [82, 'Lead', 207.21],
        'bi': [83, 'Bismuth', 208.980401],
        'po': [84, 'Polonium', 209.0],
        'at': [85, 'Astatine', 210.0],
        'rn': [86, 'Radon', 222.0],
        'fr': [87, 'Francium', 223.0],
        'ra': [88, 'Radium', 226.0],
        'ac': [89, 'Actinium', 227.0],
        'th': [90, 'Thorium', 232.03774],
        'pa': [91, 'Protactinium', 231.035881],
        'u': [92, 'Uranium', 238.028913]}

ATOM_DICT_ANUM = {1: ['h', 'Hydrogen', 1.008],
                 2: ['he', 'Helium', 4.0026022],
                 3: ['li', 'Lithium', 6.94],
                 4: ['be', 'Beryllium', 9.01218315],
                 5: ['b', 'Boron', 10.81],
                 6: ['c', 'Carbon', 12.011],
                 7: ['n', 'Nitrogen', 14.007],
                 8: ['o', 'Oxygen', 15.999],
                 9: ['f', 'Fluorine', 18.9984031635],
                 10: ['ne', 'Neon', 20.17976],
                 11: ['na', 'Sodium', 22.989769282],
                 12: ['mg', 'Magnesium', 24.305],
                 13: ['al', 'Aluminium', 26.98153843],
                 14: ['si', 'Silicon', 28.085],
                 15: ['p', 'Phosphorus', 30.9737619985],
                 16: ['s', 'Sulfur', 32.06],
                 17: ['cl', 'Chlorine', 35.45],
                 18: ['ar', 'Argon', 39.95],
                 19: ['k', 'Potassium', 39.09831],
                 20: ['ca', 'Calcium', 40.0784],
                 21: ['sc', 'Scandium', 44.9559074],
                 22: ['ti', 'Titanium', 47.8671],
                 23: ['v', 'Vanadium', 50.94151],
                 24: ['cr', 'Chromium', 51.99616],
                 25: ['mn', 'Manganese', 54.9380432],
                 26: ['fe', 'Iron', 55.8452],
                 27: ['co', 'Cobalt', 58.9331943],
                 28: ['ni', 'Nickel', 58.69344],
                 29: ['cu', 'Copper', 63.5463],
                 30: ['zn', 'Zinc', 65.382],
                 31: ['ga', 'Gallium', 69.7231],
                 32: ['ge', 'Germanium', 72.6308],
                 33: ['as', 'Arsenic', 74.9215956],
                 34: ['se', 'Selenium', 78.9718],
                 35: ['br', 'Bromine', 79.904],
                 36: ['kr', 'Krypton', 83.7982],
                 37: ['rb', 'Rubidium', 85.46783],
                 38: ['sr', 'Strontium', 87.621],
                 39: ['y', 'Yttrium', 88.9058382],
                 40: ['zr', 'Zirconium', 91.2242],
                 41: ['nb', 'Niobium', 92.906371],
                 42: ['mo', 'Molybdenum', 95.951],
                 43: ['tc', 'Technetium', 97.0],
                 44: ['ru', 'Ruthenium', 101.072],
                 45: ['rh', 'Rhodium', 102.905492],
                 46: ['pd', 'Palladium', 106.421],
                 47: ['ag', 'Silver', 107.86822],
                 48: ['cd', 'Cadmium', 112.4144],
                 49: ['in', 'Indium', 114.8181],
                 50: ['sn', 'Tin', 118.7107],
                 51: ['sb', 'Antimony', 121.7601],
                 52: ['te', 'Tellurium', 127.603],
                 53: ['i', 'Iodine', 126.904473],
                 54: ['xe', 'Xenon', 131.2936],
                 55: ['cs', 'Caesium', 132.905451966],
                 56: ['ba', 'Barium', 137.3277],
                 57: ['la', 'Lanthanum', 138.905477],
                 58: ['ce', 'Cerium', 140.1161],
                 59: ['pr', 'Praseodymium', 140.907661],
                 60: ['nd', 'Neodymium', 144.2423],
                 61: ['pm', 'Promethium', 145.0],
                 62: ['sm', 'Samarium', 150.362],
                 63: ['eu', 'Europium', 151.9641],
                 64: ['gd', 'Gadolinium', 157.253],
                 65: ['tb', 'Terbium', 158.9253547],
                 66: ['dy', 'Dysprosium', 162.5001],
                 67: ['ho', 'Holmium', 164.9303295],
                 68: ['er', 'Erbium', 167.2593],
                 69: ['tm', 'Thulium', 168.9342195],
                 70: ['yb', 'Ytterbium', 173.0451],
                 71: ['lu', 'Lutetium', 174.96681],
                 72: ['hf', 'Hafnium', 178.4866],
                 73: ['ta', 'Tantalum', 180.947882],
                 74: ['w', 'Tungsten', 183.841],
                 75: ['re', 'Rhenium', 186.2071],
                 76: ['os', 'Osmium', 190.233],
                 77: ['ir', 'Iridium', 192.2172],
                 78: ['pt', 'Platinum', 195.0849],
                 79: ['au', 'Gold', 196.9665704],
                 80: ['hg', 'Mercury', 200.5923],
                 81: ['tl', 'Thallium', 204.38],
                 82: ['pb', 'Lead', 207.21],
                 83: ['bi', 'Bismuth', 208.980401],
                 84: ['po', 'Polonium', 209.0],
                 85: ['at', 'Astatine', 210.0],
                 86: ['rn', 'Radon', 222.0],
                 87: ['fr', 'Francium', 223.0],
                 88: ['ra', 'Radium', 226.0],
                 89: ['ac', 'Actinium', 227.0],
                 90: ['th', 'Thorium', 232.03774],
                 91: ['pa', 'Protactinium', 231.035881],
                 92: ['u', 'Uranium', 238.028913]}


def read_coord_file(filename):
    """Reads turbomole coord file. Returns array of shape (5, N_atoms)."""
    datContent = np.array([i.strip().split() for i in open(filename).readlines()], dtype=object)
    coord = []
    for i in range(len(datContent)):
        if datContent[i][0].startswith("$"):
            continue
        list_entries = [float(datContent[i][j]) for j in range(3)]
        list_entries.append(datContent[i][3])
        list_entries.append(datContent[i][4] if len(datContent[i]) > 4 else "")
        coord.append(list_entries)
    return np.transpose(np.asarray(coord, dtype=object))


def write_coord_file(filename, coord, mode='w'):
    """Writes turbomole coord file."""
    with open(filename, mode) as f:
        f.write("$coord\n")
        for i in range(coord.shape[1]):
            if coord[4, i] == "":
                f.write(f"{coord[0,i]} {coord[1,i]} {coord[2,i]} {coord[3,i]}\n")
            else:
                f.write(f"{coord[0,i]} {coord[1,i]} {coord[2,i]} {coord[3,i]} {coord[4,i]}\n")
        f.write("$user-defined bonds\n$end\n")


def read_xyz_file(filename, return_header=False):
    """Reads xyz file. Returns coord_xyz of shape (4, N_atoms): [atom_type, x, y, z]."""
    datContent = [i.strip().split() for i in open(filename).readlines()]
    comment_line = np.transpose(datContent[1])
    datContent = np.array(datContent[2:], dtype=object)
    for i, item in enumerate(datContent):
        datContent[i, 1] = float(item[1])
        datContent[i, 2] = float(item[2])
        datContent[i, 3] = float(item[3])
    coord_xyz = np.transpose(datContent)
    if return_header:
        return coord_xyz, " ".join(str(x) for x in comment_line)
    return coord_xyz


def write_xyz_file(filename, coord_xyz, comment_line="", mode="w", suppress_sci_not=False):
    """Writes xyz file."""
    with open(filename, mode) as f:
        f.write(str(coord_xyz.shape[1]) + "\n")
        f.write(comment_line + "\n")
        for i in range(coord_xyz.shape[1]):
            nl = "\n" if i < coord_xyz.shape[1] - 1 else ""
            if not suppress_sci_not:
                f.write(f"{coord_xyz[0,i]}\t{coord_xyz[1,i]}\t{coord_xyz[2,i]}\t{coord_xyz[3,i]}{nl}")
            else:
                f.write(f"{coord_xyz[0,i]}\t{float(coord_xyz[1,i]):.8f}\t{float(coord_xyz[2,i]):.8f}\t{float(coord_xyz[3,i]):.8f}{nl}")
        f.write("\n")


def read_hessian(filename, n_atoms, dimensions=3):
    """Reads xtb/turbomole hessian file. Returns ndarray of shape (3*n_atoms, 3*n_atoms)."""
    datContent = [i.strip().split() for i in open(filename).readlines()[1:]]
    n = n_atoms * dimensions
    hessian = np.zeros((n, n))
    counter = 0
    if len(datContent[0]) == 7:
        lower, aoforce = 2, True
        datContent = datContent[:-1]
    elif len(datContent[0]) == 5:
        lower, aoforce = 0, False
    else:
        lower, aoforce = 0, False
    for j in range(len(datContent)):
        if aoforce:
            try:
                int(datContent[j][1])
            except ValueError:
                lower = 1
        for i in range(lower, len(datContent[j])):
            counter += 1
            row = (counter - 1) // n
            col = (counter - row * n) - 1
            try:
                hessian[row, col] = float(datContent[j][i])
            except IndexError:
                raise ValueError('n_atoms wrong. Check dimensions')
        if aoforce:
            lower = 2
    if counter != n ** 2:
        raise ValueError('n_atoms wrong. Check dimensions')
    return hessian


def shift_xyz_coord(coord_xyz, x_shift, y_shift, z_shift):
    """Shifts xyz coordinates (Angstrom)."""
    for i in range(len(coord_xyz[1, :])):
        coord_xyz[1, i] = round(float(coord_xyz[1, i]) + x_shift, 5)
        coord_xyz[2, i] = round(float(coord_xyz[2, i]) + y_shift, 5)
        coord_xyz[3, i] = round(float(coord_xyz[3, i]) + z_shift, 5)
    return coord_xyz


def shift_coord_file(coord, x_shift, y_shift, z_shift):
    """Shifts turbomole coord coordinates (Bohr)."""
    for i in range(coord.shape[1]):
        coord[0, i] += x_shift
        coord[1, i] += y_shift
        coord[2, i] += z_shift
    return coord


def align_xyz_molecule(coord_xyz, axis, molecule_axis):
    """
    Aligns a coord_xyz array along axis using atom indices in molecule_axis.
    This operates on numpy coord_xyz arrays (4, N_atoms), not torch tensors.
    """
    x_left = coord_xyz[1, molecule_axis[0]]
    y_left = coord_xyz[2, molecule_axis[0]]
    z_left = coord_xyz[3, molecule_axis[0]]
    for j in range(coord_xyz.shape[1]):
        coord_xyz[1, j] -= x_left
        coord_xyz[2, j] -= y_left
        coord_xyz[3, j] -= z_left
    x_right = coord_xyz[1, molecule_axis[1]]
    y_right = coord_xyz[2, molecule_axis[1]]
    z_right = coord_xyz[3, molecule_axis[1]]
    mol_vec = [x_right, y_right, z_right]
    angle = np.arccos(np.dot(mol_vec, axis) / (np.linalg.norm(mol_vec) * np.linalg.norm(axis)))
    if angle == 0:
        return coord_xyz
    rot_axis = np.cross(mol_vec, axis)
    rot_axis = rot_axis / np.linalg.norm(rot_axis)
    u = rot_axis
    t = angle
    R = [
        [np.cos(t) + u[0]**2*(1-np.cos(t)),       u[0]*u[1]*(1-np.cos(t)) - u[2]*np.sin(t), u[0]*u[2]*(1-np.cos(t)) + u[1]*np.sin(t)],
        [u[0]*u[1]*(1-np.cos(t)) + u[2]*np.sin(t), np.cos(t) + u[1]**2*(1-np.cos(t)),       u[1]*u[2]*(1-np.cos(t)) - u[0]*np.sin(t)],
        [u[0]*u[2]*(1-np.cos(t)) - u[1]*np.sin(t), u[1]*u[2]*(1-np.cos(t)) + u[0]*np.sin(t), np.cos(t) + u[2]**2*(1-np.cos(t))],
    ]
    for j in range(coord_xyz.shape[1]):
        v = [coord_xyz[1, j], coord_xyz[2, j], coord_xyz[3, j]]
        rv = np.asmatrix(R) * np.asmatrix(v).T
        coord_xyz[1, j] = rv[0, 0]
        coord_xyz[2, j] = rv[1, 0]
        coord_xyz[3, j] = rv[2, 0]
    return coord_xyz


gfn1_ao_num = {
    "H": 2, "He": 1,
    "Be": 4, "B": 4, "C": 4, "N": 4, "O": 4, "F": 4, "Ne": 9,
    "Li": 4, "Na": 4, "K": 4, "Rb": 4, "Cs": 4,
    "Mg": 4, "Ca": 9, "Sr": 9, "Ba": 9,
    "Al": 9, "Si": 9, "P": 9, "S": 9, "Cl": 9, "Ar": 9,
    "Ga": 9, "Ge": 9, "As": 9, "Se": 9, "Br": 9, "Kr": 9,
    "In": 9, "Sn": 9, "Sb": 9, "Te": 9, "I": 9, "Xe": 9,
    "Sc": 9, "Ti": 9, "V": 9, "Cr": 9, "Mn": 9, "Fe": 9, "Co": 9, "Ni": 9, "Cu": 9, "Zn": 4,
    "Y": 9, "Zr": 9, "Nb": 9, "Mo": 9, "Tc": 9, "Ru": 9, "Rh": 9, "Pd": 9, "Ag": 9, "Cd": 4,
    "La": 9, "Ce": 9, "Pr": 9, "Nd": 9, "Pm": 9, "Sm": 9, "Eu": 9, "Gd": 9, "Tb": 9, "Dy": 9, "Ho": 9, "Er": 9, "Tm": 9, "Yb": 9, "Lu": 9,
    "Hf": 9, "Ta": 9, "W": 9, "Re": 9, "Os": 9, "Ir": 9, "Pt": 9, "Au": 9, "Hg": 4,
    "Tl": 4, "Pb": 4, "Bi": 4, "Po": 9, "At": 9, "Rn": 9
}

gfn1_ao_num_by_ao_num = {
    0: 1,  #padding
    1: 2,  # H
    2: 1,  # He
    3: 4,  # Li
    4: 4,  # Be
    5: 4,  # B
    6: 4,  # C
    7: 4,  # N
    8: 4,  # O
    9: 4,  # F
    10: 9, # Ne
    11: 4, # Na
    12: 4, # Mg
    13: 9, # Al
    14: 9, # Si
    15: 9, # P
    16: 9, # S
    17: 9, # Cl
    18: 9, # Ar
    19: 4, # K
    20: 9, # Ca
    21: 9, # Sc
    22: 9, # Ti
    23: 9, # V
    24: 9, # Cr
    25: 9, # Mn
    26: 9, # Fe
    27: 9, # Co
    28: 9, # Ni
    29: 9, # Cu
    30: 4, # Zn
    31: 9, # Ga
    32: 9, # Ge
    33: 9, # As
    34: 9, # Se
    35: 9, # Br
    36: 9, # Kr
    37: 4, # Rb
    38: 9, # Sr
    39: 9, # Y
    40: 9, # Zr
    41: 9, # Nb
    42: 9, # Mo
    43: 9, # Tc
    44: 9, # Ru
    45: 9, # Rh
    46: 9, # Pd
    47: 9, # Ag
    48: 4, # Cd
    49: 9, # In
    50: 9, # Sn
    51: 9, # Sb
    52: 9, # Te
    53: 9, # I
    54: 9, # Xe
    55: 4, # Cs
    56: 9, # Ba
    57: 9, # La
    58: 9, # Ce
    59: 9, # Pr
    60: 9, # Nd
    61: 9, # Pm
    62: 9, # Sm
    63: 9, # Eu
    64: 9, # Gd
    65: 9, # Tb
    66: 9, # Dy
    67: 9, # Ho
    68: 9, # Er
    69: 9, # Tm
    70: 9, # Yb
    71: 9, # Lu
    72: 9, # Hf
    73: 9, # Ta
    74: 9, # W
    75: 9, # Re
    76: 9, # Os
    77: 9, # Ir
    78: 9, # Pt
    79: 9, # Au
    80: 4, # Hg
    81: 4, # Tl
    82: 4, # Pb
    83: 4, # Bi
    84: 9, # Po
    85: 9, # At
    86: 9, # Rn
}

def interpolate_energy_dependent_complex_matrix(precalculated_energy_points, matrix, new_energy):
    """
    Interpolates a complex matrix over a given energy range using PCHIP interpolation.

    Args:
        precalculated_energy_points (np.ndarray): 1D array of energy points.
        matrix (np.ndarray): 3D array of complex matrices with shape (n_e_points, dim1, dim2).
        new_energy (np.ndarray): 1D array of new energy points for interpolation.
    """
    n_e_points, dim1, dim2 = matrix.shape
    assert precalculated_energy_points.shape[0] == n_e_points, "Mismatch between precalculated energy points and matrix shape"

    if np.min(new_energy) < np.min(precalculated_energy_points):
        raise ValueError("New energy must be smaller than precalculated energy for reasonable results.")
    if np.max(new_energy) > np.max(precalculated_energy_points):
        raise ValueError("New energy must be larger than precalculated energy for reasonable results.")

    # Initialize output
    interpolated_matrix = np.empty((len(new_energy), dim1, dim2), dtype=matrix.dtype)

    # Interpolate each matrix element individually
    for i in range(dim1):
        for j in range(dim2):
            # Extract the (i, j) component over energy
            y_real = np.real(matrix[:, i, j])
            y_imag = np.imag(matrix[:, i, j])
            interpolator_real = PchipInterpolator(precalculated_energy_points, y_real)
            interpolator_imag = PchipInterpolator(precalculated_energy_points, y_imag)
            interpolated_matrix[:, i, j] = interpolator_real(new_energy) + 1j * interpolator_imag(new_energy)

    return interpolated_matrix


def read_kpoints_weights(dftb_output_file):
    """
    Reads the k-points and weights from a DFTB+ output file with a specific format.

    Args:
        dftb_output_file (str): Path to the DFTB+ output file

    Returns:
        kpoints (np.ndarray): Array of k-points (shape: [num_kpoints, 3])
        weights (np.ndarray): Array of weights (shape: [num_kpoints])
    """
    kpoints = []
    weights = []

    # Regular expression pattern to match the k-points and weights lines
    pattern = r'(\d+):\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)'

    with open(dftb_output_file, 'r') as f:
        lines = f.readlines()

    # Search for the section that contains K-points and weights
    in_kpoints_section = False
    for line in lines:
        # Detect the start of the k-points section
        if "K-points and weights:" in line:
            in_kpoints_section = True
            continue

        # If we're in the k-points section, process the lines
        if in_kpoints_section:
            # Check if we find a match for the k-point line using the regex
            match = re.match(pattern, line.strip())
            if match:
                # Extract k-point coordinates and weight from the match groups
                kpoint = np.array([float(match.group(2)), float(match.group(3)), float(match.group(4))])
                weight = float(match.group(5))

                # Append to the lists
                kpoints.append(kpoint)
                weights.append(weight)
            # If the line is empty or there's no match, stop reading
            if line.strip() == '':
                break

    # Convert lists to numpy arrays
    kpoints = np.array(kpoints)
    weights = np.array(weights)

    return kpoints, weights

def align_molecule(positions, numbers, target_axis, left_atom_indices, right_atom_indices, anchor_mode = "AuS", anchor_atom = 16):
    """
    Aligns a batch of molecules stored in positions along a specified target axis. The left and right atom indices define the molecule axis.
    For "AuS_prepare" anchor mode, molecule is aligned by the sulfur-sulfur axis, and gold atoms from the anchors are placed
    along this molecule axis. Provide sulfur indices in this case in left_atom_indices and right_atom_indices.
    Args:
        positions (torch.Tensor): Tensor of shape (B, N, 3) where B is the batch size and N is the number of atoms. If sahpe is (N, 3), a fake dimension is added.
        numbers (torch.Tensor): Tensor of shape (B, N) containing atomic numbers of the atoms in the batch. If shape is (N,), a fake dimension is added.
        target_axis (torch.Tensor or list): The axis to align the molecules along, given as a 3D vector.
        left_atom_indices (list or np.ndarray): Indices of atoms defining the left end of the molecule axis.
        right_atom_indices (list or np.ndarray): Indices of atoms defining the right end of the molecule axis.
        anchor_mode (str): Mode for handling the anchor atoms
        anchor_atom (int): Atomic number of the anchor atom (default is 16 for sulfur).
    """

    # Fake batch dimension
    if len(numbers.shape) != 2:
        numbers = numbers.unsqueeze(0)
        positions = positions.unsqueeze(0)

    device = positions.device
    #molecule_axis_indices = target_axis.to(device)
    if type(target_axis) == list or type(target_axis) == np.ndarray:
        target_axis = torch.tensor(target_axis, dtype=positions.dtype, device=device)
    target_axis = target_axis.to(device)
    B, N, _ = positions.shape

    #shift to origin
    batch_indices = torch.arange(B, device=device)
    left_points = positions[batch_indices, left_atom_indices]

    shifted_positions = positions - left_points.unsqueeze(1)

    #get molecule axis vectors
    molecule_axis_vectors = shifted_positions[batch_indices, right_atom_indices]

    v_mol = F.normalize(molecule_axis_vectors, p=2, dim=-1)
    v_target = F.normalize(target_axis.expand_as(v_mol), p=2, dim=-1)

    # Calculate angle
    dot_prod = torch.sum(v_mol * v_target, dim=-1)
    theta = torch.acos(torch.clamp(dot_prod, -1.0, 1.0))

    # Find rotation axes
    rotation_axis = torch.cross(v_mol, v_target, dim=-1)
    axis_norm = torch.linalg.norm(rotation_axis, dim=-1)
    perp_axis = torch.stack([-v_mol[:, 1], v_mol[:, 0], torch.zeros(B, device=device)], dim=1)
    perp_norm = torch.linalg.norm(perp_axis, dim=-1)
    fallback_axis = torch.tensor([1.0, 0.0, 0.0], device=device).expand_as(v_mol)
    perp_axis = torch.where(perp_norm.unsqueeze(1) < 1e-6, fallback_axis, perp_axis)

    # Update rotation axis for cases that are nearly 180 degrees
    is_anti_parallel = torch.abs(axis_norm) < 1e-6
    rotation_axis = torch.where(is_anti_parallel.unsqueeze(1), perp_axis, rotation_axis)

    # Normalize
    rotation_axis = F.normalize(rotation_axis, p=2, dim=-1)

    #Construct rotation matrices -> Rodrigues formula
    K = torch.zeros((B, 3, 3), device=device)
    K[:, 0, 1] = -rotation_axis[:, 2]
    K[:, 0, 2] = rotation_axis[:, 1]
    K[:, 1, 0] = rotation_axis[:, 2]
    K[:, 1, 2] = -rotation_axis[:, 0]
    K[:, 2, 0] = -rotation_axis[:, 1]
    K[:, 2, 1] = rotation_axis[:, 0]

    cos_theta = torch.cos(theta).unsqueeze(1).unsqueeze(2)
    sin_theta = torch.sin(theta).unsqueeze(1).unsqueeze(2)

    I = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
    u_outer = torch.bmm(rotation_axis.unsqueeze(2), rotation_axis.unsqueeze(1))
    rotation_matrix = I * cos_theta + (1 - cos_theta) * u_outer + K * sin_theta
    positions = torch.einsum('bij,bnj->bni', rotation_matrix, shifted_positions)


    if anchor_mode == "AuS_prepare":
        #make sure left and right points describe sulfur atoms
        #the indices are shifted by 1 -> see how anchors are added in GFlow_Mol
        left_atom = numbers[batch_indices, left_atom_indices]
        left_atom_pos = positions[batch_indices, left_atom_indices, :]
        right_atom = numbers[batch_indices, right_atom_indices]
        right_atom_pos = positions[batch_indices, right_atom_indices, :]
        assert torch.all(left_atom == anchor_atom), f"Left anchor atom must be Z={anchor_atom} in AuS mode."
        assert torch.all(right_atom == anchor_atom), f"Right anchor atom must be Z={anchor_atom} in AuS mode."

        #get left gold atoms
        left_atom_pos_expanded = left_atom_pos.unsqueeze(1)
        dist_vecs = positions - left_atom_pos_expanded
        dist_vecs = torch.sum(dist_vecs**2, dim = 2)
        dist_vecs_masked = torch.full_like(dist_vecs, float('inf'))
        gold_mask = (numbers == 79)
        dist_vecs_masked[gold_mask] = dist_vecs[gold_mask]
        left_gold_indices = torch.argmin(dist_vecs_masked, dim = 1)
        left_gold_atoms = numbers[batch_indices, left_gold_indices]
        assert torch.all(left_gold_atoms == 79), "Expecting gold here"
        left_gold_pos = left_atom_pos - target_axis * 2.5
        positions[batch_indices, left_gold_indices, :] = left_gold_pos

        #handle right gold atom
        right_atom_pos_expanded = right_atom_pos.unsqueeze(1)
        dist_vecs = positions - right_atom_pos_expanded
        dist_vecs = torch.sum(dist_vecs ** 2, dim=2)
        dist_vecs_masked = torch.full_like(dist_vecs, float('inf'))
        dist_vecs_masked[gold_mask] = dist_vecs[gold_mask]
        right_gold_indices = torch.argmin(dist_vecs_masked, dim=1)
        right_gold_atoms = numbers[batch_indices, right_gold_indices]
        assert torch.all(right_gold_atoms == 79), "Expecting gold here"
        right_gold_pos = right_atom_pos + target_axis * 2.5
        positions[batch_indices, right_gold_indices, :] = right_gold_pos

    return positions


def add_gold(numbers_C, positions_C, left_index, right_index, numbers_left, positions_left, numbers_right, positions_right, check_valid = False):
    """
    Adds gold tips from the left and right to the central molecule positions and numbers. Gold atoms in the central
    part are removed, and the left and right tips are translated to the positions of the anchor atoms in the central molecule.

    Args:
        numbers_C (torch.Tensor): Atomic numbers of the central molecule, shape (B, N_C).
        positions_C (torch.Tensor): Atomic positions of the central molecule, shape (B, N_C, 3).
        left_index (np.array): Index of the left anchor atom in the central molecule.
        right_index (np.array): Index of the right anchor atom in the central molecule.
        numbers_left (torch.Tensor): Atomic numbers of the left gold tip, shape (B, N_left).
        positions_left (torch.Tensor): Atomic positions of the left gold tip, shape (B, N_left, 3).
        numbers_right (torch.Tensor): Atomic numbers of the right gold tip, shape (B, N_right).
        positions_right (torch.Tensor): Atomic positions of the right gold tip, shape (B, N_right, 3).
        check_valid (bool): If True, returns a mask indicating valid geometries in the central molecule. Invalid geometries are set to exactly one zero atom

    Returns:
        numbers (torch.Tensor): Concatenated atomic numbers of the left tip, central molecule, and right tip, shape (B, N_left + N_C + N_right).
        positions (torch.Tensor): Concatenated atomic positions of the left tip, central molecule, and right tip, shape (B, N_left + N_C + N_right, 3).
        optional valid_mask (torch.Tensor): Mask indicating valid atoms in the central molecule, shape (B, N_C).
    """

    # --- Setup (same as before) ---
    device = positions_C.device
    B = positions_C.shape[0]
    batch_indices = torch.arange(B, device=device)

    left_anchor_pos = positions_C[batch_indices, left_index]
    right_anchor_pos = positions_C[batch_indices, right_index]

    translated_pos_left = positions_left.unsqueeze(0) + left_anchor_pos.unsqueeze(1)
    translated_pos_right = positions_right.unsqueeze(0) + right_anchor_pos.unsqueeze(1)

    expanded_numbers_left = numbers_left.expand(B, -1)
    expanded_numbers_right = numbers_right.expand(B, -1)

    # filter for valid atoms
    valid_mask = (numbers_C != 0) & (numbers_C != 79)
    #valid_mask = (numbers_C != 0)

    #check for unreasonable junction geometries
    if check_valid:
        threshold = 6 #todo:adapt this threshold to the system
        distances_left = torch.cdist(translated_pos_left[:, 0 : translated_pos_left.shape[1]-4, :], positions_C)
        distances_left = (distances_left > threshold).all(dim=-1).all(dim=-1)

        distances_right = torch.cdist(translated_pos_right[:, 4:, :], positions_C)
        distances_right = (distances_right > threshold).all(dim=-1).all(dim=-1)
        valid_junction_geometries = distances_left & distances_right

    processed_numbers_list = [
        torch.cat([
            expanded_numbers_left[i],
            numbers_C[i][valid_mask[i]],
            expanded_numbers_right[i]
        ]) for i in range(B)
    ]

    processed_positions_list = [
        torch.cat([
            translated_pos_left[i],
            positions_C[i][valid_mask[i]],
            translated_pos_right[i]
        ]) for i in range(B)
    ]

    #set invalid geometries to zero tensor
    if check_valid:
        #this is to handle an edge case where the largest geometry is removed -> would lead to wrong and unnecessary padding if not handeled
        zero_numbers = torch.zeros(1, dtype=numbers_C.dtype, device=device)
        zero_positions = torch.zeros((1,3), dtype=positions_C.dtype, device=device)
        processed_numbers_list = [processed_numbers_list[i] if valid_junction_geometries[i] else zero_numbers for i in range(B)]
        processed_positions_list = [processed_positions_list[i] if valid_junction_geometries[i] else zero_positions for i in range(B)]

    # padding
    numbers = mctc.batch.pack(processed_numbers_list).to(positions_C.device)
    positions = mctc.batch.pack(processed_positions_list).to(positions_C.device)

    if check_valid:
        return numbers, positions, valid_junction_geometries
    return numbers, positions

def find_anchor_atom_indices(numbers_tensor, anchor_atom=16):
    """
    Finds the indices of the left and right anchor atoms in a batch of molecules represented by a numbers tensor.

    Args:
        numbers_tensor (torch.Tensor): A tensor of shape (B, N) where B is the batch size and N is the number of atoms. If shape is (N,), a fake batch dimension is added.
        anchor_atom (int): The atomic number of the anchor atom to find. Default is 16 (sulfur).
    Returns:
        left_indices (torch.Tensor): Indices of the left anchor atoms in each molecule.
        right_indices (torch.Tensor): Indices of the right anchor atoms in each molecule.
    """

    # Fake batch dimension
    if len(numbers_tensor.shape) != 2:
        numbers_tensor = numbers_tensor.unsqueeze(0)

    batch_size = numbers_tensor.shape[0]
    # find gold indices -> where numbers tensor is anchor_atom
    anchor_indices = torch.where(numbers_tensor == anchor_atom)

    # Make sure each molecule has exactly two gold anchors
    unique_rows, counts = anchor_indices[0].unique(return_counts=True)
    assert len(unique_rows) == batch_size and torch.all(
        unique_rows == torch.arange(batch_size, device=numbers_tensor.device)) and torch.all(
        counts == 2), "Anchors seem to be wrong. For this method to work, each molecule must have exactly two anchor atoms. Do some fragments contain sulfur?"

    # all even indices are left, all odd indices are right
    left_indices = anchor_indices[1][::2]
    right_indices = anchor_indices[1][1::2]

    return left_indices, right_indices

def print_vram_usage():
    """Prints the total, free, and used VRAM in megabytes."""
    if torch.cuda.is_available():
        # Get the total and free memory in bytes
        # .mem_get_info() returns (free, total)
        free, total = torch.cuda.mem_get_info()

        # Convert bytes to megabytes (MB)
        total_mb = total / 1024 ** 2
        free_mb = free / 1024 ** 2
        used_mb = (total - free) / 1024 ** 2

        print(f"Total VRAM: {total_mb:.2f} MB")
        print(f"Used VRAM:  {used_mb:.2f} MB")
        print(f"Free VRAM:  {free_mb:.2f} MB")
    else:
        print("CUDA is not available. VRAM usage cannot be displayed.")


def append_batch_to_xyz(filepath, numbers, positions, hash_values, batch_indices, worker_id, stage):
    """
    Manually appends a batch of molecules to a single XYZ file.
    Holds the file open for the duration of the batch to prevent filesystem stress.
    """
    pt = Chem.GetPeriodicTable()
    
    # Ensure tensors are on CPU and numpy-compatible
    if hasattr(numbers, "cpu"): numbers = numbers.cpu()
    if hasattr(positions, "cpu"): positions = positions.cpu()

    # convert to angstrom
    positions = positions.numpy() / ANG2BOHR

    with open(filepath, "a") as f:
        counter = 0
        for i in range(len(hash_values)):
            if i in batch_indices:

                mol_nums = numbers[counter]
                mol_pos = positions[counter]
                
                # filter padded atoms
                mask = mol_nums != 0
                real_nums = mol_nums[mask]
                real_pos = mol_pos[mask]
                
                # header
                f.write(f"{len(real_nums)}\n")
                f.write(f"hash={hash_values[i]} stage={stage} batch_idx={i} worker_id={worker_id}\n")
                

                for z, r in zip(real_nums, real_pos):
                    z_int = int(z.item()) if hasattr(z, "item") else int(z)
                    r_list = r.tolist() if hasattr(r, "tolist") else r
                    
                    symbol = pt.GetElementSymbol(z_int)
                    f.write(f"{symbol:<2} {r_list[0]:12.8f} {r_list[1]:12.8f} {r_list[2]:12.8f}\n")
                
                counter += 1

def append_to_hdf5(filepath, group_name, data_dict, attributes={}):
    """
    Appends data to a single HDF5 file.
    Args:
        filepath: Path to the .h5 file.
        group_name: The group path (e.g., 'hash_12345/phonon').
        data_dict: Dictionary of datasets {name: numpy_array}.
        attributes: Dictionary of metadata attributes.
    """
    # Open in 'a' (append) mode. Safe for single-process access.
    with h5py.File(filepath, 'a') as f:
        # Create group if it doesn't exist
        if group_name not in f:
            grp = f.create_group(group_name)
        else:
            grp = f[group_name]

        # Write datasets
        for key, data in data_dict.items():
            if key in grp:
                del grp[key]  # Overwrite if exists
            grp.create_dataset(key, data=data)

        # Write attributes (metadata)
        for key, val in attributes.items():
            grp.attrs[key] = val

def fig_to_numpy(fig):
    """Converts a matplotlib figure to a numpy array (RGB) for storage."""
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='png', dpi=100)
    io_buf.seek(0)
    return np.frombuffer(io_buf.getvalue(), dtype=np.uint8)

# --- Welzl's Algorithm for smallest Enclosing Circle ---
# https://en.wikipedia.org/wiki/Smallest-circle_problem
# https://www.geeksforgeeks.org/dsa/minimum-enclosing-circle-using-welzls-algorithm/
def get_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def is_in_circle(p, circle):
    c, r = circle
    return get_distance(p, c) <= r + 1e-9

def get_circle_with_2_points(p1, p2):
    center_x = (p1[0] + p2[0]) / 2
    center_y = (p1[1] + p2[1]) / 2
    radius = get_distance(p1, p2) / 2
    return ((center_x, center_y), radius)

def get_circle_with_3_points(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    D = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(D) < 1e-9:
        d12 = get_distance(p1, p2)
        d13 = get_distance(p1, p3)
        d23 = get_distance(p2, p3)
        if d12 >= d13 and d12 >= d23: return get_circle_with_2_points(p1, p2)
        if d13 >= d12 and d13 >= d23: return get_circle_with_2_points(p1, p3)
        return get_circle_with_2_points(p2, p3)
    center_x = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / D
    center_y = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / D
    radius = get_distance((center_x, center_y), p1)
    return ((center_x, center_y), radius)

def welzl(P, R):
    if not P or len(R) == 3:
        if len(R) == 0: return ((0, 0), 0)
        elif len(R) == 1: return (R[0], 0)
        elif len(R) == 2: return get_circle_with_2_points(R[0], R[1])
        else: return get_circle_with_3_points(R[0], R[1], R[2])
    p = P[0]
    D = welzl(P[1:], R)
    if is_in_circle(p, D):
        return D
    return welzl(P[1:], R + [p])

def make_circle(points):
    P = list(points)
    random.shuffle(P)
    return welzl(P, [])

if __name__ == '__main__':
    pass




