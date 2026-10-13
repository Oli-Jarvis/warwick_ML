import numpy as np
from ase.io import read
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import re


def read_kpoints_weights(dftb_output_file):
    """
    Reads k-points and weights from a DFTB+ output file, including the first line.

    Parameters:
        dftb_output_file (str): Path to the DFTB+ output file

    Returns:
        kpoints (np.ndarray): Array of shape (N_kpoints, 3)
        weights (np.ndarray): Array of shape (N_kpoints,)
    """
    kpoints = []
    weights = []

    # Regex pattern to extract: index, kx, ky, kz, weight
    pattern = re.compile(
        r"\s*(\d+):\s+([-+]?\d*\.\d+)\s+([-+]?\d*\.\d+)\s+([-+]?\d*\.\d+)\s+([-+]?\d*\.\d+)"
    )

    with open(dftb_output_file, 'r') as f:
        for line in f:
            if "K-points and weights:" in line:
                # Try to extract k-point directly from this same line
                match = pattern.search(line)
                if match:
                    kpoint = [float(match.group(i)) for i in range(2, 5)]
                    weight = float(match.group(5))
                    kpoints.append(kpoint)
                    weights.append(weight)
                continue  # proceed to next lines

            # Match any further k-point lines
            match = pattern.match(line)
            if match:
                kpoint = [float(match.group(i)) for i in range(2, 5)]
                weight = float(match.group(5))
                kpoints.append(kpoint)
                weights.append(weight)

    return np.array(kpoints), np.array(weights)

def read_dftbplus_matrices(path, N_k_vec: int, N_orb: int) -> tuple:
    """
    Reads the Hamiltonian and overlap matrices from the DFTB+ output files.

    Args:
        path (str): Path to the directory containing the DFTB+ output files.
        N_k_vec (int): Number of k-points.
        N_orb (int): Number of orbitals.
    Returns:
        tuple: Hamiltonian and overlap matrices in k-space.
    Note:
        This is not very efficient, but this is done only once and the matrices are stored.
    """
    h_k = np.zeros((N_k_vec, N_orb, N_orb), dtype=complex)
    s_k = np.zeros((N_k_vec, N_orb, N_orb), dtype=complex)
    h_k_filename = f"{path}/hamsqr1.dat"
    s_k_filename = f"{path}/oversqr.dat"
    for n in range(N_k_vec):
        print(n)
        # hamiltonian
        part = np.loadtxt(h_k_filename, skiprows=5 + n * (N_orb + 3), dtype=float, max_rows=N_orb)
        real_part = part[:, 0::2]
        imag_part = part[:, 1::2]
        part = real_part + 1.j * imag_part
        print(part.shape)
        H_01_k = part[part.shape[0] // 2:, 0: part.shape[0] // 2]
        H_10_k = part[0: part.shape[0] // 2, part.shape[0] // 2:]
        assert np.allclose(H_01_k, np.conj(H_10_k.T)), "Error in Hamiltonian matrix, most likely the two pricipal layers are not set up properly."
        h_k[n, :, :] = part

        # overlap matrix
        part = np.loadtxt(s_k_filename, skiprows=5 + n * (N_orb + 3), dtype=float, max_rows=N_orb)
        real_part = part[:, 0::2]
        imag_part = part[:, 1::2]
        part = real_part + 1.j * imag_part
        S_01_k = part[part.shape[0] // 2:, 0: part.shape[0] // 2]
        S_10_k = part[0: part.shape[0] // 2, part.shape[0] // 2:]
        assert np.allclose(S_01_k, np.conj(S_10_k.T))
        s_k[n, :, :] = part
    return h_k, s_k

def read_dftbplus_matrices_fast(filepath, N_k_vec: int, N_orb: int) -> tuple:
    """
    Reads the Hamiltonian and overlap matrices from the DFTB+ output files.

    Args:
        filepath (str): Path to  DFTB+ matrix files.
        N_k_vec (int): Number of k-points.
        N_orb (int): Number of orbitals.
    Returns:
        tuple: read matrix in k-space.
    """

    def _read_file(filepath):
        with open(filepath, encoding='utf-8') as file:
            for line in file:
                yield line

    m_k = np.zeros((N_k_vec, N_orb, N_orb), dtype=complex)

    k_vec_processed = 0
    buffer = []
    skip_part = True
    skip_counter = 0
    for i, line in enumerate(_read_file(filepath)):
        if i < 2:
            continue

        if line.startswith("#") or (skip_part and skip_counter < 3):
            #print(line)
            skip_part = True
            skip_counter += 1
            continue

        else:
            skip_part = False
            skip_counter = 0

            temp = line.strip().split()
            #cast to float
            temp = [float(i) for i in temp]
            buffer.extend(temp)
            if len(buffer) == N_orb*2*N_orb:
                real_part = np.array(buffer[0::2])
                imag_part = np.array(buffer[1::2])
                part = real_part + 1.j * imag_part
                part = np.reshape(part, (N_orb, N_orb))
                #print(part.shape)
                m_01_k = part[part.shape[0] // 2:, 0: part.shape[0] // 2]
                m_10_k = part[0: part.shape[0] // 2, part.shape[0] // 2:]
                assert np.allclose(m_01_k, np.conj(m_10_k.T)), f"Error in hermitian matrix from {filepath}, most likely the two principal layers are not set up properly."
                m_k[k_vec_processed, :, :] = part

                #reset everything
                buffer = []
                print(k_vec_processed)
                k_vec_processed += 1

    return m_k


def discrete_fourier_transform(matrix: np.ndarray, N_k_vec: int, dftb_output_file = None) -> np.ndarray:
    """
    Performs discrete fourier transform on the input matrix. Averaging is enough here, because the Fourier transform was
    done regarding the super-cell lattice vectors. Placing two principal layers in a super-cell allows us to transform
    back to the real space and then do the partitioning. Since we concentrate on the supercell at the origin the back-
    transform is just an average over the k-vectors.

    Args:
        matrix (np.ndarray): Input matrix to be transformed.
        N_k_vec (int): Number of k-points.
        dftb_output_file (str): Path to the DFTB+ output file -> if given, the k-points weights are read from the file and used in the average

    Returns:
        np.ndarray: Transformed matrix.

    References:
        Pauly, F., Viljas, J. K., Huniar, U., Häfner, M., Wohlthat, S., Bürkle, M., ... & Schön, G. (2008). Cluster-based density-functional approach to quantum transport through molecular and atomic contacts. New Journal of Physics, 10(12), 125019.
        Ji, X., Qi, Q., Chen, Y., Zhou, C., & Yu, X. (2025). A Three-Tiered Hierarchical Computational Framework Bridging Molecular Systems and Junction-Level Charge Transport. Journal of Chemical Theory and Computation, 21(6), 2961-2976.

    """
    if dftb_output_file is not None:
        kpoints, weights = read_kpoints_weights(dftb_output_file)
        N_k_vec = len(kpoints)
        matrix = np.average(matrix, axis=0, weights=weights)
        return matrix

    matrix = 1/N_k_vec*np.sum(matrix, axis=0)
    return matrix

def decimation(E: float, H_00: np.ndarray, H_01: np.ndarray, S_00: np.ndarray, S_01: np.ndarray, eta = 1E-6, max_iter=200, eps = 1e-12) -> tuple:
    """
    Constructs the surface and bulk Green's function using the decimation method.
    Args:
        E (float): Energy at which to compute the Green's function.
        H_00 (np.ndarray): Hamiltonian matrix for principal layer (should contain two pricipal layers, oriented along z).
        H_01 (np.ndarray): Hamiltonian matrix coupling between two principal layers.
        S_00 (np.ndarray): Overlap matrix -> see H_00.
        S_01 (np.ndarray): Overlap matrix -> see H_01.
        eta (float): Small imaginary part to avoid singularities.
        max_iter (int): Maximum number of iterations for convergence.
        eps (float): Convergence criterion.

    Returns:
        G_surf (np.ndarray): Surface Green's function.
        G_bulk (np.ndarray): Bulk Green's function.

    References:
        Pauly, F., Viljas, J. K., Huniar, U., Häfner, M., Wohlthat, S., Bürkle, M., ... & Schön, G. (2008). Cluster-based density-functional approach to quantum transport through molecular and atomic contacts. New Journal of Physics, 10(12), 125019.
        Ji, X., Qi, Q., Chen, Y., Zhou, C., & Yu, X. (2025). A Three-Tiered Hierarchical Computational Framework Bridging Molecular Systems and Junction-Level Charge Transport. Journal of Chemical Theory and Computation, 21(6), 2961-2976.

    """

    W = (E+eta*1.j)*np.identity(H_00.shape[0], dtype=complex)@S_00-H_00
    tau_1 = (E+eta*1.j)*np.identity(H_00.shape[0], dtype=complex)@S_01 - H_01
    tau_2 = (E+eta*1.j)*np.identity(H_00.shape[0], dtype=complex)@np.conj(np.transpose(S_01)) - np.conj(np.transpose(H_01))
    W_b = W
    W_s = W
    converged = False
    for i in range(max_iter):
        W_b_inv = np.linalg.inv(W_b)
        temp = tau_1@W_b_inv@tau_2
        W_s = W_s - temp
        W_b = W_b - temp - tau_2@W_b_inv@tau_1
        tau_1 = -tau_1@ W_b_inv @tau_1
        tau_2 = -tau_2 @ W_b_inv @ tau_2

        delta = np.abs(np.sum(tau_1@W_b_inv@tau_2))
        if delta < eps:
            converged = True
            break
    if not(converged):
        #todo: This has to be handled better.
        raise ValueError("Not converged")
        #pass

    G_surf = np.linalg.inv(W_s)
    G_bulk = np.linalg.inv(W_b)
    return G_surf, G_bulk


if __name__ == '__main__':

    N_orb = 648
    N_k_vec = 516
    #using high number of threads works best for OMP_NUM_THREADS=1.
    num_threads = 32
    #Energy range in eV
    E_min = -18
    E_max = -6
    N_E_points = 1000
    #Fermi level in eV
    #E_f = -8.1816

    path = "FILL_IN"
    print("string to read DFTB+ matrices")

    dftb_output_file = f"{path}/log.out"

    h_k_filepath = f"{path}/hamsqr1.dat"
    #h_k, s_k = read_dftbplus_matrices(path, N_k_vec, N_orb)
    h_k = read_dftbplus_matrices_fast(h_k_filepath, N_k_vec, N_orb)
    H = discrete_fourier_transform(h_k, N_k_vec, dftb_output_file)
    s_k_filepath = f"{path}/oversqr.dat"
    s_k = read_dftbplus_matrices_fast(s_k_filepath, N_k_vec, N_orb)
    S = discrete_fourier_transform(s_k, N_k_vec, dftb_output_file)

    H_00 = H[0 : H.shape[0]//2, 0 : H.shape[0]//2]
    H_01 = H[0 : H.shape[0]//2:, H.shape[0]//2:]

    S_00 = S[0 : S.shape[0]//2, 0 : S.shape[0]//2]
    S_01 = S[0 : S.shape[0]//2:, S.shape[0]//2:]

    np.savetxt(f"{path}/H_00.dat", H_00)
    np.savetxt(f"{path}/S_00.dat", S_00)

    E = np.linspace(E_min, E_max, N_E_points)
    Ha2eV = 27.21138602
    E = E/Ha2eV
    G_funcs = []
    def decimation_handling(e):
        G_surf, G_bulk = decimation(e, H_00, H_01, S_00, S_01, eta=1E-2, max_iter=500, eps=1e-6)
        return G_surf, G_bulk

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(decimation_handling, e): e for e in E}
        for future in tqdm(as_completed(futures), total=len(E)):
            G_funcs.append(future.result())

    G_funcs = np.array(G_funcs)
    G_surf  = G_funcs[:,0,:,:]
    G_bulk = G_funcs[:, 1, :, :]

    def calculate_dos(G):
        dos = -(1/np.pi)*np.imag(np.trace(G, axis1=1, axis2=2))
        return dos

    dos_surf = calculate_dos(G_surf)
    dos_bulk = calculate_dos(G_bulk)

    G_surf.tofile(f"{path}/G_surf")
    np.savetxt(f"{path}/dos_surf.dat", (E,dos_surf), header="E (Ha), DOS_surf (1/Ha)")
    np.savetxt(f"{path}/dos_bulk.dat", (E,dos_bulk), header="E (Ha), DOS_bulk (1/Ha)")

    plt.plot(E*Ha2eV, dos_surf*(1/Ha2eV), label = "DOS_surf")
    plt.plot(E*Ha2eV, dos_bulk*(1/Ha2eV), label = "DOS_bulk")
    #plt.axvline(E_f, color='b')
    plt.xlabel("E (eV)")
    plt.legend()
    plt.ylabel("DOS (1/eV)")
    plt.savefig(f"{path}/DOS.pdf")