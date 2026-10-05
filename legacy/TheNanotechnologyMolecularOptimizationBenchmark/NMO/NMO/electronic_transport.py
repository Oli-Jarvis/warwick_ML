import numpy as np
import glob

import torch
from ase.io import read
import os
from .utils import interpolate_energy_dependent_complex_matrix, gfn1_ao_num_by_ao_num


from .constants import __Ha2eV__, __e0__, __hP__, __G0__



data_type_complex = torch.complex64
data_type_real = torch.float

class Electronic_Transport_Calculator:
    """
    Class to calculate the transport properties of a molecule connected to two leads.
    """
    def __init__(self, H_ECC, S_ECC, E_min, E_max, N_E_points, g_surf_left_path, g_surf_right_path, coord_xyz_ecc,
                 T=300, WBL = False, strict = False):

        self.H_ECC = H_ECC
        self.S_ECC = S_ECC
        self.E_min = E_min
        self.E_max = E_max
        self.N_E_points = N_E_points
        self.g_surf_left_path = g_surf_left_path
        self.g_surf_right_path = g_surf_right_path
        self.coord_xyz_ecc = coord_xyz_ecc
        #Temperature in Kelvin
        self.T = T
        #use wide band limit
        self.WBL = WBL
        self.strict = strict

        self._g_surf_left = None
        self._g_surf_right = None
        self._E = None
        self._E_fermi_shift = None
        self._E_fermi = None
        self._N_orb_lead = None
        self._tau_el = None
        self._K_0 = None
        self._K_1 = None
        self._K_2 = None
        self._G_el = None
        self._S_el = None
        self._kappa_el = None

        if self.strict:
            self.check_system()

        #check if this is called from a parent class
        if type(self) is Electronic_Transport_Calculator:
            self.N_orb_junction = self.H_ECC.shape[0] - 2 * self.N_orb_lead//2
            self.left_range = [0, self.N_orb_lead//2]
            self.center_range = [self.N_orb_lead//2, self.N_orb_lead//2+self.N_orb_junction]
            self.right_range = [self.N_orb_lead//2+self.N_orb_junction, self.N_orb_junction+self.N_orb_lead//2*2]

            self.H_CC = self.H_ECC[self.N_orb_lead // 2:self.N_orb_lead // 2 + self.N_orb_junction,
                        self.N_orb_lead // 2:self.N_orb_lead // 2 + self.N_orb_junction]
            self.S_CC = self.S_ECC[self.N_orb_lead // 2:self.N_orb_lead // 2 + self.N_orb_junction,
                        self.N_orb_lead // 2:self.N_orb_lead // 2 + self.N_orb_junction]

            self.set_E_fermi_shift()




    @property
    def g_surf_left(self):
        if self._g_surf_left is None:
            self._g_surf_left = self.load_surface_gf(self.g_surf_left_path)
            return self._g_surf_left
        else:
            return self._g_surf_left

    @g_surf_left.setter
    def g_surf_left(self, value):
        self._g_surf_left = value

    @property
    def g_surf_right(self):
        if self._g_surf_right is None:
            self._g_surf_right = self.load_surface_gf(self.g_surf_right_path)
            return self._g_surf_right
        else:
            return self._g_surf_right

    @g_surf_right.setter
    def g_surf_right(self, value):
        self._g_surf_right = value

    @property
    def E(self):
        """
        Energy range in Ha
        :return:
        """
        if self._E is None:
            self._E = np.linspace(self.E_min, self.E_max, self.N_E_points) / __Ha2eV__
        return self._E

    @property
    def E_fermi_shift(self):
        if self._E_fermi_shift is None:
            self._E_fermi_shift = self.set_E_fermi_shift()
        return self._E_fermi_shift

    @property
    def E_fermi(self):
        """
        Fermi energy in eV
        :return:
        """
        if self._E_fermi is None:
            self._E_fermi = self.get_E_fermi()
        return self._E_fermi

    @property
    def N_orb_lead(self):
        if self._N_orb_lead is None:
            self._N_orb_lead = self.get_N_orb_lead()
        return self._N_orb_lead

    @property
    def tau_el(self):
        if self._tau_el is None:
            self._tau_el = self.calculate_tau_el()
        return self._tau_el

    @property
    def K_0(self):
        if self._K_0 is None:
            self._K_0 = self.calculate_K_n(0)
        return self._K_0

    @property
    def K_1(self):
        if self._K_1 is None:
            self._K_1 = self.calculate_K_n(1)
        return self._K_1

    @property
    def K_2(self):
        if self._K_2 is None:
            self._K_2 = self.calculate_K_n(2)
        return self._K_2

    @property
    def G_el(self):
        """
        Calculate the G_el in units of G_0 at the Fermi energy.
        """
        if self._G_el is None:
            self._G_el = self.calculate_G_el()
        return self._G_el

    @property
    def S_el(self):
        """
        Thermopower in units of muV/K
        """
        if self._S_el is None:
            self._S_el = self.calculate_S_el()
        return self._S_el

    @property
    def kappa_el(self):
        """
        Electronic contribution to the thermal conductance in pW/K
        """
        if self._kappa_el is None:
            self._kappa_el = self.calculate_kappa_el()
        return self._kappa_el

    def get_E_fermi(self):
        """
        Reads the Fermi energy from the lead calculations

        Returns:
            float: Fermi energy in eV
        """

        def get_fermi_level_eV(filename):
            with open(filename, 'r') as file:
                for line in file:
                    if line.startswith("Fermi level"):
                        parts = line.split()
                        # The value in eV is the last part of the line
                        fermi_level_eV = float(parts[-2])  # second to last is the number, last is "eV"
                        return fermi_level_eV
            raise ValueError("Fermi level not found in file.")
        E_f_left = get_fermi_level_eV(f"{self.g_surf_left_path}/detailed_first.out")
        E_f_right = get_fermi_level_eV(f"{self.g_surf_right_path}/detailed_first.out")
        if abs(E_f_left - E_f_right) > 1:
            raise ValueError("Fermi levels of left and right lead have siginificant difference")
        return (E_f_left+E_f_right)/2

    def get_N_orb_lead(self):
        """
        Reads the number of orbitals from the lead calculations

        Returns:
            int: Number of orbitals in the lead
        """
        H_00_bulk_left = np.loadtxt(f"{self.g_surf_left_path}/H_00.dat", dtype=complex)
        H_00_bulk_right = np.loadtxt(f"{self.g_surf_right_path}/H_00.dat", dtype=complex)
        #Double it because H_00 is only one principal layer
        N_orb_lead_left = H_00_bulk_left.shape[0]*2
        N_orb_lead_right = H_00_bulk_right.shape[0]*2
        if N_orb_lead_left != N_orb_lead_right:
            raise ValueError("Number of orbitals in left and right lead are not equal. Not implemented yet")
        return N_orb_lead_left



    def set_E_fermi_shift(self):
        H_00_bulk_left = np.loadtxt(f"{self.g_surf_left_path}/H_00.dat", dtype=complex)
        H_00_bulk_right = np.loadtxt(f"{self.g_surf_right_path}/H_00.dat", dtype=complex)

        #The following lines are an alternative way to calculate the shift. It is not used in the current version.
        """
        N_orb_au = 9
        eigvals_bulk_left = []
        eigvals_bulk_right = []
        eigvals_finite_left = []
        eigvals_finite_right = []
        for i in range(H_00_bulk_left.shape[0]//9):
            H_cut = H_00_bulk_left[i*N_orb_au:(i+1)*N_orb_au, i*N_orb_au:(i+1)*N_orb_au]
            eigenvalues, eigenvectors = np.linalg.eigh(H_cut)
            eigenvalues = sorted(eigenvalues)
            eigvals_bulk_left.append(eigenvalues[0])
            #take smallest eigenvalue
        for i in range(H_00_bulk_right.shape[0]//9):
            H_cut = H_00_bulk_right[i*N_orb_au:(i+1)*N_orb_au, i*N_orb_au:(i+1)*N_orb_au]
            eigenvalues, eigenvectors = np.linalg.eigh(H_cut)
            eigenvalues = sorted(eigenvalues)
            eigvals_bulk_right.append(eigenvalues[0])
        for i in range(H_00_bulk_left.shape[0]//9):
            H_cut = self.H_ECC[i*N_orb_au:(i+1)*N_orb_au, i*N_orb_au:(i+1)*N_orb_au]
            eigenvalues, eigenvectors = np.linalg.eigh(H_cut)
            eigenvalues = sorted(eigenvalues)
            eigvals_finite_left.append(eigenvalues[0])
        for i in range(H_00_bulk_left.shape[0]//9):
            H_cut = self.H_ECC[self.left_range[0]+i*N_orb_au:(i+1)*N_orb_au, self.left_range[0]+i*N_orb_au:(i+1)*N_orb_au]
            eigenvalues, eigenvectors = np.linalg.eigh(H_cut)
            eigenvalues = sorted(eigenvalues)
            eigvals_finite_right.append(eigenvalues[0])
        #return 0

        shift_left = (np.mean(eigvals_bulk_left)-np.mean(eigvals_finite_left))*__Ha2eV__
        shift_right = (np.mean(eigvals_bulk_right)-np.mean(eigvals_finite_right))*__Ha2eV__
        print("optimal shift", shift_left, shift_right)
        """

        #calculate shift according to
        #Verzijl, C. J. O., & Thijssen, J. M. (2012). DFT-based molecular transport implementation in ADF/BAND. The Journal of Physical Chemistry C, 116(46), 24393-24412.
        #Ji, X., Qi, Q., Chen, Y., Zhou, C., & Yu, X. (2025). A Three-Tiered Hierarchical Computational Framework Bridging Molecular Systems and Junction-Level Charge Transport. Journal of Chemical Theory and Computation, 21(6), 2961-2976.

        shift_left = 0
        for k in range(H_00_bulk_left.shape[0]):
            shift_left += (H_00_bulk_left[k,k] - self.H_ECC[self.left_range[0]+k, self.left_range[0]+k]) / self.S_ECC[self.left_range[0]+k, self.left_range[0]+k]
        shift_right = 0
        for k in range(H_00_bulk_right.shape[0]):
            shift_right += (H_00_bulk_right[k,k] - self.H_ECC[self.right_range[0]+k, self.right_range[0]+k]) / self.S_ECC[self.right_range[0]+k, self.right_range[0]+k]

        shift_left = shift_left / (H_00_bulk_left.shape[0])
        shift_right = shift_right / (H_00_bulk_right.shape[0])
        shift = np.real((shift_left + shift_right))/2

        #shifting finite calculation to bulk calculation
        #Verzijl, C. J. O., & Thijssen, J. M. (2012). DFT-based molecular transport implementation in ADF/BAND. The Journal of Physical Chemistry C, 116(46), 24393-24412.
        self.H_ECC = self.H_ECC + shift * self.S_ECC

        self.H_CC = self.H_ECC[self.N_orb_lead // 2:self.N_orb_lead // 2 + self.N_orb_junction,
                    self.N_orb_lead // 2:self.N_orb_lead // 2 + self.N_orb_junction]
        self.S_CC = self.S_ECC[self.N_orb_lead // 2:self.N_orb_lead // 2 + self.N_orb_junction,
                    self.N_orb_lead // 2:self.N_orb_lead // 2 + self.N_orb_junction]

        return shift



    def load_surface_gf(self, path):
        # Load the surface Green's function from the specified file
        if not self.WBL:
            g_surf = np.fromfile(f"{path}/G_surf")
            g_surf = g_surf[0::2] + 1.j * g_surf[1::2]

            #load energy points
            dos_surf = np.loadtxt(f"{path}/dos_surf.dat", skiprows=1)
            calculated_E_points = dos_surf[0, :]
            g_surf = np.reshape(g_surf, (calculated_E_points.shape[0], self.N_orb_lead//2, self.N_orb_lead//2))

            #check if interpolation is needed
            if calculated_E_points.shape[0] == self.E.shape[0] and np.allclose(calculated_E_points, self.E):
                return g_surf
                pass
            else:
                print("Interpolation needed for g_surf. Consider to recalculate g_surf because interpolation is slow")

            g_surf = interpolate_energy_dependent_complex_matrix(calculated_E_points, g_surf, self.E)
            return g_surf
        else:
            g_surf = np.zeros((self.N_E_points, self.N_orb_lead//2, self.N_orb_lead//2), dtype=complex)
            for i in range(g_surf.shape[0]):
                for j in range(g_surf.shape[1]):
                    #in 1/Ha
                    g_surf[i, j, j] = 1.j*np.pi*0.036*__Ha2eV__
            return g_surf
            


    def calculate_tau_el(self):

        def compute_Sigma(H_XC, S_XC, g_XX, E):
            """
            Computes Sigma according to:
            Sigma(E) = (H_CX - E S_CX) * g_XX(E) * (H_XC - E S_XC)

            All matrices have shape (N_E, N_orb, N_orb)
            energies has shape (N_E,)
            """
            H_CX = np.conj(np.transpose(H_XC))
            S_CX = np.conj(np.transpose(S_XC))

            # Compute (H_CX - E S_CX) and (H_XC - E S_XC)
            H_CX_eff = H_CX - E[:, np.newaxis, np.newaxis] * S_CX
            H_XC_eff = H_XC - E[:, np.newaxis, np.newaxis] * S_XC

            # First multiplication: (H_CX_eff @ g_XX)
            temp = np.matmul(H_CX_eff, g_XX)

            # Second multiplication: (temp @ H_XC_eff)
            Sigma = np.matmul(temp, H_XC_eff)

            return Sigma


        H_CL = self.H_ECC[self.left_range[0]:self.left_range[1], self.center_range[0]:self.center_range[1]]
        S_CL = self.S_ECC[self.left_range[0]:self.left_range[1], self.center_range[0]:self.center_range[1]]
        Sigma_L = compute_Sigma(H_CL, S_CL, self.g_surf_left, self.E)
        Gamma_L =-2. * np.imag(Sigma_L)

        H_CR = self.H_ECC[self.right_range[0]:self.right_range[1], self.center_range[0]:self.center_range[1]]
        S_CR = self.S_ECC[self.right_range[0]:self.right_range[1], self.center_range[0]:self.center_range[1]]
        Sigma_R = compute_Sigma(H_CR, S_CR, self.g_surf_right, self.E)
        Gamma_R = -2. * np.imag(Sigma_R)

        eta = 1E-6
        G_r_CC = np.linalg.inv(self.E[:, np.newaxis, np.newaxis] * (1 + 1.j * eta) * self.S_CC - self.H_CC - Sigma_L - Sigma_R)
        G_a_CC = np.conj(np.transpose(G_r_CC, (0, 2, 1)))
        tau = np.real(np.trace(np.matmul(np.matmul(Gamma_L, G_r_CC), np.matmul(Gamma_R, G_a_CC)), axis1=1, axis2=2))

        return tau


    def calculate_K_n(self, n, mu=-1):
        """
        Calculate the K_n according to:
        Bürkle, M., Hellmuth, T. J., Pauly, F., & Asai, Y. (2015). First-principles calculation of the thermoelectric
        figure of merit for [2, 2] paracyclophane-based single-molecule junctions. Physical Review B, 91(16), 165419.
        Equation (6)

        Args:
            n (int): Order of the K_n. 0 for G_el, 1 for K_1, etc.
            mu (float): Chemical potential in eV. Default is -1, which means that the chemical potential is the Fermi energy.
        """
        if mu == -1:
            mu = self.E_fermi
        def _dfermi(mu):
            kB = 8.6173303e-5  # eV/K
            beta = 1.0 / (kB * self.T)  # 1/eV
            x = beta * (self.E*__Ha2eV__ - mu)
            return -np.exp(x) * beta / (np.exp(x) + 1.0) ** 2

        df = _dfermi(mu)
        intg = -self.tau_el * df * (self.E*__Ha2eV__ - mu) ** n
        integral = np.trapz(intg, self.E*__Ha2eV__)
        return integral

    def calculate_G_el(self, mu = -1):
        """
        Calculate the G_el in units of G_0 according to:
        Bürkle, M., Hellmuth, T. J., Pauly, F., & Asai, Y. (2015). First-principles calculation of the thermoelectric
        figure of merit for [2, 2] paracyclophane-based single-molecule junctions. Physical Review B, 91(16), 165419.
        Equation (3).

        Args:
            mu (float): Chemical potential in eV. Default is -1, which means that the chemical potential is the Fermi energy.

        Returns:
            G_el (float): Conductance in units of G_0.
        """
        if mu == -1:
            return self.K_0
        else:
            # Calculate the conductance at the given chemical potential
            return self.calculate_K_n(0, mu)

    def calculate_S_el(self, mu = -1):
        """
        Calculate the S_el in units of muV/K according to:
        Bürkle, M., Hellmuth, T. J., Pauly, F., & Asai, Y. (2015). First-principles calculation of the thermoelectric
        figure of merit for [2, 2] paracyclophane-based single-molecule junctions. Physical Review B, 91(16), 165419.
        Equation (4).

        Args:
            mu (float): Chemical potential in eV. Default is -1, which means that the chemical potential is the Fermi energy.

        Returns:
            S_el (float): Thermopower in units of muV/K.
        """
        if mu == -1:
            K_0 = self.K_0
            K_1 = self.K_1
        else:
            K_0 = self.calculate_K_n(0, mu)
            K_1 = self.calculate_K_n(1, mu)

        S = -K_1 / (__e0__*self.T * K_0)
        S *= __e0__ * 1e6  # μV/K
        return S

    def calculate_kappa_el(self, mu = -1):
        """
        Calculate the S_el in units of muV/K according to:
        Bürkle, M., Hellmuth, T. J., Pauly, F., & Asai, Y. (2015). First-principles calculation of the thermoelectric
        figure of merit for [2, 2] paracyclophane-based single-molecule junctions. Physical Review B, 91(16), 165419.
        Equation (5).

        Args:
            mu (float): Chemical potential in eV. Default is -1, which means that the chemical potential is the Fermi energy.

        Returns:
            S_el (float): Thermopower in units of muV/K.
        """
        if mu == -1:
            K_0 = self.K_0
            K_1 = self.K_1
            K_2 = self.K_2
        else:
            K_0 = self.calculate_K_n(0, mu)
            K_1 = self.calculate_K_n(1, mu)
            K_2 = self.calculate_K_n(2, mu)

        kappa_el = 2.0 / (__hP__ * self.T) * (K_2 - K_1 ** 2 / K_0)
        kappa_el *= __e0__ * 1e12  # pW/K

        return kappa_el




    def check_system(self):
        """
        Check if the system is set up correctly. This includes checking if the lead calculations are properly positioned
        with the ecc coordinates.
        """

        def check_lead_positions(path, which_lead):
            """
            Check if atoms in the lead calculations are properly positioned with the ecc coordinates.
            Allowed is a relative shift

            Args:
                path (str): Path to the lead calculation
                which_lead (str): "left" or "right"
            """
            poscar_file = glob.glob(os.path.join(path, '*.poscar'))
            if not poscar_file:
                raise FileNotFoundError(f"No POSCAR file found in the given path: {path}")
            poscar_file = poscar_file[0]
            atoms = read(poscar_file)
            positions = atoms.get_positions()

            if which_lead == "left":
                ecc_starting_pos = 0
            elif which_lead == "right":
                ecc_starting_pos = self.coord_xyz_ecc.shape[1]-positions.shape[0]//2
            else:
                raise ValueError(f"Unknown lead: {which_lead}")

            global_shift = self.coord_xyz_ecc[1:,ecc_starting_pos] - positions[0,:]
            #check if the atoms are in the right position
            for i in range(positions.shape[0]//2):
                if np.linalg.norm(self.coord_xyz_ecc[1:,i+ecc_starting_pos] - positions[i,:] - global_shift) > 1E-4:
                    raise ValueError(f"wrong position of atoms in lead or ecc: {path}")

        check_lead_positions(self.g_surf_left_path, "left")
        check_lead_positions(self.g_surf_right_path, "right")



class Electronic_Transport_Calculator_torch(Electronic_Transport_Calculator):
    """
    Class to calculate the transport properties of a molecule connected to two leads.
    """

    import torch

    def __init__(self, el_structure_calculator, E_min, E_max, N_E_points, g_surf_left_path, g_surf_right_path, T=300, WBL = False):

        self.dd = {"dtype": data_type_real, "device": torch.device("cpu")}
        if torch.cuda.is_available():
            self.dd = {"dtype": data_type_real, "device": torch.device("cuda:0")}
            print("cuda")


        self.el_structure_calculator = el_structure_calculator

        emo_torch = el_structure_calculator.mo_energies
        Cs_torch = el_structure_calculator.coefficients
        S_torch = el_structure_calculator.overlap
        self.S_ECC = S_torch

        emo_diag_mat = torch.diag_embed(emo_torch)  # (B, N, N)
        self.H_ECC = torch.matmul(
            torch.matmul(
                torch.matmul(
                    torch.matmul(S_torch, Cs_torch),
                    emo_diag_mat
                ),
                Cs_torch.transpose(-2, -1)
            ),
            S_torch
        )
        if len(self.H_ECC.shape) == 3:
            self.batch_size = self.H_ECC.shape[0]
        else:
            #introduce fake batching
            self.H_ECC = self.H_ECC.unsqueeze(0)
            self.S_ECC = self.S_ECC.unsqueeze(0)
            self.batch_size = 1


        H_00_bulk_left = np.loadtxt(f"{g_surf_left_path}/H_00.dat", dtype=complex)
        S_00_bulk_left = np.loadtxt(f"{g_surf_left_path}/S_00.dat", dtype=complex)
        H_00_bulk_right = np.loadtxt(f"{g_surf_right_path}/H_00.dat", dtype=complex)
        S_00_bulk_right = np.loadtxt(f"{g_surf_right_path}/S_00.dat", dtype=complex)

        self.H_00_bulk_left = torch.tensor(H_00_bulk_left, **self.dd)
        self.S_00_bulk_left = torch.tensor(S_00_bulk_left, **self.dd)
        self.H_00_bulk_right = torch.tensor(H_00_bulk_right, **self.dd)
        self.S_00_bulk_right = torch.tensor(S_00_bulk_right, **self.dd)

        #call the parent constructor
        super().__init__(self.H_ECC, self.S_ECC, E_min, E_max, N_E_points,
                         g_surf_left_path, g_surf_right_path, coord_xyz_ecc = None,
                         T=T, WBL=WBL, strict=False)

        N_orb_lead = torch.tensor(self.N_orb_lead).unsqueeze(0).expand(self.batch_size)
        N_orb_junction = self.H_ECC.shape[1] - 2 * (N_orb_lead // 2)

        self.left_range = torch.stack([torch.zeros(self.batch_size, dtype=torch.long), N_orb_lead // 2], dim=1)
        self.center_range = torch.stack([N_orb_lead // 2, N_orb_lead // 2 + N_orb_junction], dim=1)
        self.right_range = torch.stack([N_orb_lead // 2 + N_orb_junction, N_orb_lead // 2 + N_orb_junction + N_orb_lead // 2], dim=1)


        #partition the system
        H_CL_batch = []
        S_CL_batch = []
        H_CR_batch = []
        S_CR_batch = []
        H_CC_batch = []
        S_CC_batch = []
        H_LL_batch = []
        S_LL_batch = []
        H_RR_batch = []
        S_RR_batch = []

        for b in range(self.batch_size):
            l0, l1 = self.left_range[b].tolist()
            c0, c1 = self.center_range[b].tolist()
            r0, r1 = self.right_range[b].tolist()

            H_CL_batch.append(self.H_ECC[b, l0:l1, c0:c1])
            S_CL_batch.append(self.S_ECC[b, l0:l1, c0:c1])
            H_CR_batch.append(self.H_ECC[b, r0:r1, c0:c1])
            S_CR_batch.append(self.S_ECC[b, r0:r1, c0:c1])
            H_CC_batch.append(self.H_ECC[b, c0:c1, c0:c1])
            S_CC_batch.append(self.S_ECC[b, c0:c1, c0:c1])
            H_LL_batch.append(self.H_ECC[b, l0:l1, l0:l1])
            S_LL_batch.append(self.S_ECC[b, l0:l1, l0:l1])
            H_RR_batch.append(self.H_ECC[b, r0:r1, r0:r1])
            S_RR_batch.append(self.S_ECC[b, r0:r1, r0:r1])

        self.H_CL = torch.stack(H_CL_batch, dim=0)
        self.S_CL = torch.stack(S_CL_batch, dim=0)

        self.H_CR = torch.stack(H_CR_batch, dim=0)
        self.S_CR = torch.stack(S_CR_batch, dim=0)

        self.H_CC = torch.stack(H_CC_batch, dim=0)
        self.S_CC = torch.stack(S_CC_batch, dim=0)

        self.H_LL = torch.stack(H_LL_batch, dim=0)
        self.S_LL = torch.stack(S_LL_batch, dim=0)

        self.H_RR = torch.stack(H_RR_batch, dim=0)
        self.S_RR = torch.stack(S_RR_batch, dim=0)

        #try deliting ECC marices
        del self.H_ECC
        del self.S_ECC
        torch.cuda.empty_cache()



        self.set_E_fermi_shift()


    @property
    def E(self):
        """
        Energy range in Ha
        :return:
        """
        if self._E is None:
            self._E = torch.linspace(self.E_min, self.E_max, self.N_E_points, **self.dd) / __Ha2eV__
        return self._E


    def set_E_fermi_shift(self):
        """
        Set the E_fermi_shift
        """
        #calculate shift according to
        #Verzijl, C. J. O., & Thijssen, J. M. (2012). DFT-based molecular transport implementation in ADF/BAND. The Journal of Physical Chemistry C, 116(46), 24393-24412.
        #Ji, X., Qi, Q., Chen, Y., Zhou, C., & Yu, X. (2025). A Three-Tiered Hierarchical Computational Framework Bridging Molecular Systems and Junction-Level Charge Transport. Journal of Chemical Theory and Computation, 21(6), 2961-2976.
        #get left shift
        H_00_bulk_left_diagonal = torch.diagonal(self.H_00_bulk_left).unsqueeze(0)
        S_00_bulk_left_diagonal = torch.diagonal(self.S_00_bulk_left).unsqueeze(0)
        H_LL_diagonal = torch.diagonal(self.H_LL, dim1=-2, dim2=-1)
        shift_left = 1/H_00_bulk_left_diagonal.shape[-1]*((H_00_bulk_left_diagonal-H_LL_diagonal)/S_00_bulk_left_diagonal).sum(-1)

        #get right shift
        H_00_bulk_right_diagonal = torch.diagonal(self.H_00_bulk_right).unsqueeze(0)
        S_00_bulk_right_diagonal = torch.diagonal(self.S_00_bulk_right).unsqueeze(0)
        H_RR_diagonal = torch.diagonal(self.H_RR, dim1=-2, dim2=-1)
        shift_right = 1/H_00_bulk_right_diagonal.shape[-1]*((H_00_bulk_right_diagonal-H_RR_diagonal)/S_00_bulk_right_diagonal).sum(-1)

        #mean of left and right shift
        shift = (shift_left + shift_right) / 2.0

        #shifting finite calculation to bulk calculation
        #Verzijl, C. J. O., & Thijssen, J. M. (2012). DFT-based molecular transport implementation in ADF/BAND. The Journal of Physical Chemistry C, 116(46), 24393-24412.
        self.H_CL = self.H_CL + shift.unsqueeze(-1).unsqueeze(-1) * self.S_CL
        self.H_CC = self.H_CC + shift.unsqueeze(-1).unsqueeze(-1) * self.S_CC
        self.H_CR = self.H_CR + shift.unsqueeze(-1).unsqueeze(-1) * self.S_CR




    def load_surface_gf(self, path):
        # Load the surface Green's function from the specified file
        if not self.WBL:
            g_surf = np.fromfile(f"{path}/G_surf")
            g_surf = g_surf[0::2] + 1.j * g_surf[1::2]

            # load energy points
            dos_surf = np.loadtxt(f"{path}/dos_surf.dat", skiprows=1)
            calculated_E_points = dos_surf[0, :]
            g_surf = np.reshape(g_surf, (calculated_E_points.shape[-1], self.N_orb_lead // 2, self.N_orb_lead // 2))
            dd = self.dd
            dd["dtype"] = data_type_complex
            g_surf = torch.tensor(g_surf, **dd)

            # check if interpolation is needed
            print("interpolation check missing")
            """
            if calculated_E_points.shape[0] == self.E.shape[0] and torch.allclose(calculated_E_points, self.E):
                return g_surf
                pass
            else:
                raise ValueError("Energy points do not match. Interpolation is needed. Not supported for torch implementation -> not performant")
            """
            return g_surf
        else:
            g_surf = np.zeros((self.N_E_points, self.N_orb_lead // 2, self.N_orb_lead // 2), dtype=complex)
            for i in range(g_surf.shape[0]):
                for j in range(g_surf.shape[1]):
                    # in 1/Ha
                    g_surf[i, j, j] = 1.j * np.pi * 0.036 * __Ha2eV__
            dd = self.dd
            dd["dtype"] = data_type_complex
            g_surf = torch.tensor(g_surf, **dd)
            return g_surf


    def calculate_tau_el_old(self):

        def compute_Sigma(H_XC, S_XC, g_XX, E):
            """
            Computes Sigma according to:
            Sigma(E) = (H_CX - E S_CX) * g_XX(E) * (H_XC - E S_XC)

            All matrices have shape (N_E, N_orb, N_orb)
            energies has shape (N_E,)
            """

            #H_CX = torch.conj(H_XC.transpose(-2, -1))
            #S_CX = torch.conj(S_XC.transpose(-2, -1))

            #H_CX_eff = (H_CX - E_exp * S_CX).to(torch.complex128)
            #H_XC_eff = (H_XC - E_exp * S_XC).to(torch.complex128)
            g_XX_exp = g_XX.to(data_type_complex)
            Sigma = torch.empty(self.batch_size, g_XX_exp.shape[1], S_XC.shape[-1], S_XC.shape[-1], device=S_XC.device, dtype=g_XX_exp.dtype)
            #loop over batch size and batch over energy points. Usually the batch size is small and the energy points are large
            for i in range(self.batch_size):
                H_CX_i = torch.conj(H_XC[i].transpose(-2, -1))
                S_CX_i = torch.conj(S_XC[i].transpose(-2, -1))
                H_CX_eff = (H_CX_i - E_exp[0] * S_CX_i).to(data_type_complex)
                H_XC_eff = (H_XC[i] - E_exp[0] * S_XC[i]).to(data_type_complex)
                Sigma[i] = torch.bmm(torch.bmm(H_CX_eff, g_XX_exp[0]), H_XC_eff)

            return Sigma

        #Compute self-energies
        E_exp = self.E.view(1, -1, 1, 1)
        #left
        H_CL = self.H_CL.unsqueeze(1)
        S_CL = self.S_CL.unsqueeze(1)
        g_surf_left = self.g_surf_left.unsqueeze(0)
        Sigma_L = compute_Sigma(H_CL, S_CL, g_surf_left, E_exp)
        #right
        H_CR = self.H_CR.unsqueeze(1)
        S_CR = self.S_CR.unsqueeze(1)
        g_surf_right = self.g_surf_right.unsqueeze(0)
        Sigma_R = compute_Sigma(H_CR, S_CR, g_surf_right, E_exp)

        #Compute Gamma
        Gamma_L = (-2. * torch.imag(Sigma_L)).to(data_type_complex)
        Gamma_R = (-2. * torch.imag(Sigma_R)).to(data_type_complex)

        #Greens function
        eta = 1e-6
        E_mat = self.E[:, None, None] * (1 + 1.j * eta)
        E_mat = E_mat.unsqueeze(0)  # shape (1, N_E, 1, 1)

        S_CC = self.S_CC.to(data_type_complex)
        H_CC = self.H_CC.to(data_type_complex)

        S_CC = S_CC.unsqueeze(1)
        H_CC = H_CC.unsqueeze(1)

        #loop over batch size and batch over energy points. Usually the batch size is small and the energy points are large
        tau = torch.empty(self.batch_size, E_mat.shape[1], device=E_mat.device, dtype=E_mat.dtype)
        for i in range(self.batch_size):
            G_r_CC = torch.linalg.inv(E_mat[0] * S_CC[i] - H_CC[i] - Sigma_L[i] - Sigma_R[i])
            G_a_CC = torch.conj(G_r_CC.transpose(-2, -1))
            mat = torch.bmm(torch.bmm(Gamma_L[i], G_r_CC), torch.bmm(Gamma_R[i], G_a_CC))
            trace = torch.diagonal(mat, dim1=-2, dim2=-1).sum(-1)
            tau[i] = torch.real(trace)

        return tau

    def calculate_tau_el(self):

        def compute_Sigma_batch(H_XC_batch, S_XC_batch, g_XX_all_energies, E_all_energies):
            """
            Computes Sigma for a single batch of H_XC and S_XC across all energies.
            Sigma(E) = (H_CX - E S_CX) * g_XX(E) * (H_XC - E S_XC)

            H_XC_batch, S_XC_batch have shape (N_orb, N_orb) (for a single batch element)
            g_XX_all_energies has shape (N_E, N_orb, N_orb)
            E_all_energies has shape (N_E,)

            Returns Sigma with shape (N_E, N_orb, N_orb)
            """

            # Add energy and batch dimensions for broadcasting
            E_exp = E_all_energies.view(-1, 1, 1).to(data_type_complex)

            H_XC_exp = H_XC_batch.unsqueeze(0).to(data_type_complex)
            S_XC_exp = S_XC_batch.unsqueeze(0).to(data_type_complex)

            H_CX_exp = torch.conj(H_XC_exp.transpose(-2, -1))
            S_CX_exp = torch.conj(S_XC_exp.transpose(-2, -1))

            # operations will broadcast over the energy dimension
            H_CX_eff = H_CX_exp - E_exp * S_CX_exp
            H_XC_eff = H_XC_exp - E_exp * S_XC_exp

            g_XX_exp = g_XX_all_energies.to(data_type_complex)

            # Perform batched matrix multiplications over the energy dimension
            Sigma = torch.bmm(torch.bmm(H_CX_eff, g_XX_exp), H_XC_eff)

            return Sigma

        # Store results for all batches
        tau_all_batches = torch.empty(self.batch_size, self.E.shape[0], device=self.dd["device"], dtype=self.E.dtype)

        # Iterate over each batch of the primary data
        for i in range(self.batch_size):
            # Compute self-energies for the current batch element
            # left
            H_CL_i = self.H_CL[i] # Shape (N_orb, N_orb)
            S_CL_i = self.S_CL[i] # Shape (N_orb, N_orb)
            # g_surf_left is already over all energies, shape (N_E, N_orb, N_orb)
            Sigma_L_i = compute_Sigma_batch(H_CL_i, S_CL_i, self.g_surf_left, self.E)

            # right
            H_CR_i = self.H_CR[i] # Shape (N_orb, N_orb)
            S_CR_i = self.S_CR[i] # Shape (N_orb, N_orb)
            # g_surf_right is already over all energies, shape (N_E, N_orb, N_orb)
            Sigma_R_i = compute_Sigma_batch(H_CR_i, S_CR_i, self.g_surf_right, self.E)

            # Compute Gamma for the current batch element
            Gamma_L_i = (-2. * torch.imag(Sigma_L_i)).to(data_type_complex)
            Gamma_R_i = (-2. * torch.imag(Sigma_R_i)).to(data_type_complex)

            # Greens function for the current batch element
            eta = 1e-6
            E_mat = self.E.view(-1, 1, 1) * (1 + 1.j * eta) # Shape (N_E, 1, 1)

            S_CC_i = self.S_CC[i].unsqueeze(0).to(data_type_complex)
            H_CC_i = self.H_CC[i].unsqueeze(0).to(data_type_complex)

            # operations will broadcast over the energy dimension
            matrix_to_invert = E_mat * S_CC_i - H_CC_i - Sigma_L_i - Sigma_R_i

            # Compute inverse using batch matrix inverse
            G_r_CC_i, info = torch.linalg.inv_ex(matrix_to_invert, check_errors=False)
            invalid_mask = (info != 0)
            if torch.any(invalid_mask):
                tau_all_batches[i] = torch.zeros(matrix_to_invert.shape[0],
                                                 dtype=torch.float64,
                                                 device=matrix_to_invert.device)
                print("Matrix inversion failed for some energy points in batch ", i)
            else:
                G_a_CC_i = torch.conj(G_r_CC_i.transpose(-2, -1))

                del Sigma_L_i
                del Sigma_R_i

                # Compute transmission tau for the current batch element across all energies
                #mat_i = torch.bmm(torch.bmm(Gamma_L_i, G_r_CC_i), torch.bmm(Gamma_R_i, G_a_CC_i)) # Shape (N_E, N_orb, N_orb)
                mat_i = torch.einsum('bij,bjk,bkl,blm->bim', Gamma_L_i, G_r_CC_i, Gamma_R_i, G_a_CC_i)

                # Trace and take the real part
                trace_i = torch.diagonal(mat_i, dim1=-2, dim2=-1).sum(-1)
                tau_i = torch.real(trace_i)

                tau_all_batches[i] = tau_i

        return tau_all_batches

    def calculate_K_n(self, n, mu=-1):
        """
        Calculate the K_n according to:
        Bürkle, M., Hellmuth, T. J., Pauly, F., & Asai, Y. (2015). First-principles calculation of the thermoelectric
        figure of merit for [2, 2] paracyclophane-based single-molecule junctions. Physical Review B, 91(16), 165419.
        Equation (6)

        Args:
            n (int): Order of the K_n. 0 for G_el, 1 for K_1, etc.
            mu (float): Chemical potential in eV. Default is -1, which means that the chemical potential is the Fermi energy.
        """
        if mu == -1:
            mu = self.E_fermi
        def _dfermi(mu):
            kB = 8.6173303e-5  # eV/K
            beta = 1.0 / (kB * self.T)  # 1/eV
            x = beta * (self.E*__Ha2eV__ - mu)

            return -torch.exp(x) * beta / (torch.exp(x) + 1.0) ** 2

        df = _dfermi(mu)
        #replace nan and inf with 0
        df = torch.nan_to_num(df, nan=0.0, posinf=0.0, neginf=0.0)
        intg = -self.tau_el * df * (self.E*__Ha2eV__ - mu) ** n
        integral = torch.trapz(intg, self.E*__Ha2eV__)

        return integral


class Electronic_Transport_Estimator_torch(Electronic_Transport_Calculator):
    """
    Class to calculate the transport properties of a molecule connected to two leads. In contrast to the
    Electronic_Transport_Calculator, this class does not require the full ECC matrices, but only the center part for faster
    evaluation. Works only with WBL limit.
    """

    def __init__(self, el_structure_calculator, E_min, E_max, N_E_points, anchor_atom = 79, T=300):

        import torch
        self.dd = {"dtype": data_type_real, "device": torch.device("cpu")}
        if torch.cuda.is_available():
            self.dd = {"dtype": data_type_real, "device": torch.device("cuda:0")}
            print("cuda")

        self.el_structure_calculator = el_structure_calculator

        emo_torch = el_structure_calculator.mo_energies
        Cs_torch = el_structure_calculator.coefficients
        S_torch = el_structure_calculator.overlap
        self.S_CC = S_torch

        emo_diag_mat = torch.diag_embed(emo_torch)  # (B, N, N)
        self.H_CC = torch.matmul(
            torch.matmul(
                torch.matmul(
                    torch.matmul(S_torch, Cs_torch),
                    emo_diag_mat
                ),
                Cs_torch.transpose(-2, -1)
            ),
            S_torch
        )

        self.batch_size = self.H_CC.shape[0]

        #find ranges of left_index and right_index
        numbers_tensor = self.el_structure_calculator.numbers
        #find gold indices -> where numbers tensor is anchor_atom
        gold_indices = torch.where(numbers_tensor == anchor_atom)

        #Make sure each molecule has exactly two gold anchors
        unique_rows, counts = gold_indices[0].unique(return_counts=True)
        assert len(unique_rows) == self.batch_size and torch.all(
            unique_rows == torch.arange(self.batch_size, device=numbers_tensor.device)) and torch.all(
            counts == 2), "Gold anchors seem to be wrong"

        #all even indices are left, all odd indices are right
        self.left_indices = gold_indices[1][::2]
        self.right_indices = gold_indices[1][1::2]

        numbers_np = numbers_tensor.cpu().numpy()
        ranges_list = [[gfn1_ao_num_by_ao_num[num] for num in numbers_np[i]] for i in range(numbers_tensor.shape[0])]
        ranges = np.cumsum(np.array(ranges_list), axis=1, dtype=np.int64)
        ranges = np.insert(ranges, 0, 0, axis=1)
        self.left_range = np.array([ranges[i, self.left_indices[i] : self.left_indices[i] + 2] for i in range(self.batch_size)])
        self.right_range = np.array([ranges[i, self.right_indices[i] : self.right_indices[i] + 2] for i in range(self.batch_size)])

        #call the parent constructor
        super().__init__(self.H_CC, self.S_CC, E_min, E_max, N_E_points, None, None, None,
                         T=T, WBL=True, strict=False)


    @property
    def E(self):
        """
        Energy range in Ha
        :return:
        """
        if self._E is None:
            self._E = torch.linspace(self.E_min, self.E_max, self.N_E_points, **self.dd) / __Ha2eV__
        return self._E

    @property
    def E_fermi(self):
        """
        Fermi energy in eV
        :return:
        """
        if self._E_fermi is None:
            self._E_fermi = self.get_E_fermi()
        return self._E_fermi

    def get_E_fermi(self):
        """
        Get the Fermi energy from E_lumo - E_homo/2
        :return:
        """

        #see where occupation is > 0
        occ_a = self.el_structure_calculator.occupation[:,0]
        occ_b = self.el_structure_calculator.occupation[:,1]
        #check if occupation is different along axis 1
        open_shell = torch.any(occ_a != occ_b, dim=1)
        if torch.any(open_shell):
            raise ValueError("Cannot handle open shell calculations")
        occ = occ_a
        num_levels = occ.shape[1]
        is_occupied = occ > 0
        flipped_mask = torch.flip(is_occupied, dims=[1])
        homo_indices = (num_levels - 1) - torch.argmax(flipped_mask.int(), dim=1)
        lumo_indices = (num_levels ) - torch.argmax(flipped_mask.int(), dim=1)

        energies = self.el_structure_calculator.mo_energies
        e_homo = torch.gather(energies, 1, homo_indices.unsqueeze(1)).squeeze(1)
        e_lumo = torch.gather(energies, 1, lumo_indices.unsqueeze(1)).squeeze(1)
        fermi_energies = (e_homo + e_lumo) / 2

        return fermi_energies


    def calculate_tau_el(self):

        # Store results for all batches
        tau_all_batches = torch.empty(self.batch_size, self.E.shape[0], device=self.dd["device"], dtype=self.E.dtype)
        device = tau_all_batches.device

        #handle zero padded stuff
        diagonal_mask = torch.diagonal(self.H_CC, dim1=-2, dim2=-1) != 0
        valid_sizes = torch.sum(diagonal_mask, dim=1)
        rows_and_cols = torch.arange(self.H_CC[0].size(-1), device=self.dd["device"])
        block_mask = ((rows_and_cols[None, :, None] < valid_sizes[:, None, None]) &
                      (rows_and_cols[None, None, :] < valid_sizes[:, None, None]))
        identity_batch = torch.eye(self.H_CC[0].size(-1), device=self.dd["device"], dtype=data_type_complex).expand(self.batch_size, -1, -1)

        # Iterate over batch
        for i in range(self.batch_size):

            # Compute self-energies for the current batch element
            Sigma_L_i = torch.zeros(self.H_CC.size(-1), self.H_CC.size(-1), dtype=data_type_complex, device=device)
            diag_indices = np.arange(self.left_range[i, 0], self.left_range[i, 1])
            Sigma_L_i[diag_indices, diag_indices] = -0.1 / __Ha2eV__ * 1.j
            Sigma_L_i = Sigma_L_i.unsqueeze(0)

            Sigma_R_i = torch.zeros(self.H_CC.size(-1), self.H_CC.size(-1), dtype=data_type_complex, device=device)
            diag_indices = np.arange(self.right_range[i, 0], self.right_range[i, 1])
            Sigma_R_i[diag_indices, diag_indices] = -0.1 / __Ha2eV__ * 1.j
            Sigma_R_i = Sigma_R_i.unsqueeze(0)

            Gamma_L_i = (-2. * torch.imag(Sigma_L_i)).to(data_type_complex) # Shape (N_E, N_orb, N_orb)
            Gamma_R_i = (-2. * torch.imag(Sigma_R_i)).to(data_type_complex) # Shape (N_E, N_orb, N_orb)

            # Greens function for the current batch element
            eta = 1e-6
            E_mat = self.E.view(-1, 1, 1) * (1 + 1.j * eta) # Shape (N_E, 1, 1)

            S_CC_i = self.S_CC[i].unsqueeze(0).to(data_type_complex) # Shape (1, N_orb, N_orb)
            H_CC_i = self.H_CC[i].unsqueeze(0).to(data_type_complex) # Shape (1, N_orb, N_orb)

            # Broadcast over the energy dimension
            matrix_to_invert = E_mat * S_CC_i - H_CC_i - Sigma_L_i - Sigma_R_i
            #handle zero padded stuff
            matrix_to_invert = torch.where(block_mask[i], matrix_to_invert, identity_batch[i])
            G_r_CC_i = torch.linalg.solve(matrix_to_invert, identity_batch[i])

            # Compute inverse using batch matrix inverse
            #G_r_CC_i = torch.linalg.inv(matrix_to_invert) # Shape (N_E, N_orb, N_orb)
            G_a_CC_i = torch.conj(G_r_CC_i.transpose(-2, -1)) # Shape (N_E, N_orb, N_orb)

            del Sigma_L_i
            del Sigma_R_i

            # Compute transmission tau for the current batch element across all energies
            Gamma_L_i = Gamma_L_i.expand(self.E.size(0), -1, -1)
            Gamma_R_i = Gamma_R_i.expand(self.E.size(0), -1, -1)

            #mat_i = torch.bmm(torch.bmm(Gamma_L_i, G_r_CC_i), torch.bmm(Gamma_R_i, G_a_CC_i)) # Shape (N_E, N_orb, N_orb)
            mat_i = torch.einsum('bij,bjk,bkl,blm->bim', Gamma_L_i, G_r_CC_i, Gamma_R_i, G_a_CC_i)

            # Trace and take the real part
            trace_i = torch.diagonal(mat_i, dim1=-2, dim2=-1).sum(-1) # Shape (N_E,)
            tau_i = torch.real(trace_i) # Shape (N_E,)

            # Store the result for the current batch element
            tau_all_batches[i] = tau_i


        return tau_all_batches



    def calculate_K_n(self, n, mu=-1):
        """
        Calculate the K_n according to:
        Bürkle, M., Hellmuth, T. J., Pauly, F., & Asai, Y. (2015). First-principles calculation of the thermoelectric
        figure of merit for [2, 2] paracyclophane-based single-molecule junctions. Physical Review B, 91(16), 165419.
        Equation (6)

        Args:
            n (int): Order of the K_n. 0 for G_el, 1 for K_1, etc.
            mu (float): Chemical potential in eV. Default is -1, which means that the chemical potential is the Fermi energy.
        """
        if mu == -1:
            mu = self.E_fermi
        def _dfermi(mu):
            kB = 8.6173303e-5  # eV/K
            beta = 1.0 / (kB * self.T)  # 1/eV
            x = beta * (self.E*__Ha2eV__ - mu.view(-1, 1))

            return -torch.exp(x) * beta / (torch.exp(x) + 1.0) ** 2

        df = _dfermi(mu)
        intg = -self.tau_el * df * (self.E*__Ha2eV__ - mu.view(-1, 1)) ** n
        integral = torch.trapz(intg, self.E*__Ha2eV__)

        return integral




if __name__ == '__main__':
    pass





