import numpy as np
import glob

import torch
from ase.io import read
import os

from .utils import interpolate_energy_dependent_complex_matrix, gfn1_ao_num_by_ao_num
import scipy.signal


from .constants import __Ha2eV__, __e0__, __hP__, __G0__, __conversion__, __h_bar__, __w2J__, __ang2bohr__, __eV2Ha__, __k_B__, __har2pJ__, __h__
from .utils import ATOM_DICT_ANUM


data_type_complex = torch.complex64
data_type_real = torch.float

class Phononic_Transport_Estimator_torch():
    """
    Class to calculate the phononic transport properties of a isolated molecule similar to arXiv:2505.19158.This class
    does not require/support the full ECC matrices, but only the center part for faster
    evaluation. Implements a Debye model for the electrodes
    """

    def __init__(self, el_structure_calculator, E_D, N_E_points, gamma = -7, anchor_atom = 79, T=300, hessian = None):

        self.dd = {"dtype": data_type_real, "device": torch.device("cpu")}
        if torch.cuda.is_available():
            self.dd = {"dtype": data_type_real, "device": torch.device("cuda")}
            print("cuda")

        self.el_structure_calculator = el_structure_calculator
        if hessian is None:
            self.hessian = self.el_structure_calculator.hessian
        else:
            self.hessian = hessian
        #self.hessian = hessian
        self.batch_size = self.hessian.shape[0]
        self.E_D = E_D
        self.N_E_points = N_E_points
        self.gamma = gamma
        self.T = T

        #system checks
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
        self.left_indices = gold_indices[1][::2].cpu().detach().numpy()
        self.right_indices = gold_indices[1][1::2].cpu().detach().numpy()

        #todo: Improve this. works for the current use cases, but is not general
        M_C = int(numbers_tensor[0,self.left_indices[0]].cpu().detach().numpy())
        self.M_C = ATOM_DICT_ANUM[M_C][2]

        self._w = None
        self._E = None
        self._w_D = None
        self._g0 = None
        self._sigma = None
        self._tau_ph = None
        self._kappa_el = None

    @property
    @torch.no_grad()
    def w(self):
        """
        frequency range
        :return:
        """
        if self._w is None:

            self._w = torch.linspace(0, self.w_D*1.1, self.N_E_points, **self.dd)
        return self._w

    @property
    @torch.no_grad()
    def E(self):
        """
        energy in meV
        :return:
        """
        if self._E is None:
            self._E = self.w*np.sqrt(__conversion__)*__h_bar__/(__w2J__)
        return self._E
    

    @property
    @torch.no_grad()
    def w_D(self):
        if self._w_D is None:
            # convert to J
            E_D = self.E_D * __w2J__
            # convert to 1/s
            w_D = E_D / __h_bar__
            w_D = w_D / np.sqrt(__conversion__)
            self._w_D = w_D
        return self._w_D


    @property
    @torch.no_grad()
    def g0(self):
        #use scipy implementation here as it has to be calculated only once and the implementation is trusted
        if self._g0 is None:
            def im_g(w):
                if (w <= w_D_np):
                    Im_g = -np.pi * 3.0 * w / (2 * w_D_np ** 3)
                else:
                    Im_g = 0
                return Im_g
            w_np = self.w.cpu().numpy()
            w_D_np = self.w_D
            Im_g = map(im_g, w_np)
            Im_g = np.asarray(list(Im_g))
            Re_g = -np.asarray(np.imag(scipy.signal.hilbert(Im_g)))
            g0 = np.asarray((Re_g + 1.j * Im_g), complex)
            #transform to torch tensor
            g0 = torch.tensor(g0, device=self.w.device, dtype=data_type_complex)
            self._g0 = g0
        return self._g0

    @property
    @torch.no_grad()
    def sigma(self):
        if self._sigma is None:

            gamma_hb = self.gamma * (__eV2Ha__/__ang2bohr__**2)
            #todo: do not hard code this
            M_L =  self.M_C
            M_C =  self.M_C

            gamma_prime = gamma_hb/np.sqrt(M_L * M_C)

            g = self.g0/(1+gamma_prime*self.g0)
            sigma = gamma_prime**2 * g

            self._sigma = sigma
        return self._sigma

    @property
    @torch.no_grad()
    def tau_ph(self):
        """
        Calculate the phonon transmission.
        :return:
        """
        if self._tau_ph is None:
            tau_ph = self.calculate_tau_ph()
            self._tau_ph = tau_ph
        return self._tau_ph

    @property
    @torch.no_grad()
    def kappa_ph(self):
        """
        Calculate the thermal conductance in pW/K.
        :return:
        """
        if self._kappa_ph is None:
            #calculate kappa_el according to Mingo 2006
            kappa_el = self.calculate_kappa_ph()
            self._kappa_el = kappa_el
        return self._kappa_el

    def calculate_tau_ph_old(self):

        # Store results for all batches
        tau_all_batches = torch.empty(self.batch_size, self.w.shape[0], device=self.dd["device"], dtype=self.w.dtype)
        device = tau_all_batches.device

        #handle zero padded stuff
        diagonal_mask = torch.diagonal(self.hessian, dim1=-2, dim2=-1) != 0
        valid_sizes = torch.sum(diagonal_mask, dim=1)
        rows_and_cols = torch.arange(self.hessian[0].size(-1), device=self.dd["device"])
        block_mask = ((rows_and_cols[None, :, None] < valid_sizes[:, None, None]) &
                      (rows_and_cols[None, None, :] < valid_sizes[:, None, None]))
        identity_batch = torch.eye(self.hessian[0].size(-1), device=self.dd["device"], dtype=data_type_complex).expand(self.batch_size, -1, -1)

        # Iterate over batch
        for i in range(self.batch_size):

            # Compute self-energies for the current batch element
            Sigma_L_i = torch.zeros(self.N_E_points, self.hessian.size(-1), self.hessian.size(-1), dtype=data_type_complex, device=device)
            left_index = self.left_indices[i]
            matrix = torch.eye(3, dtype=data_type_complex, device = device) * self.sigma[:, None, None]
            Sigma_L_i[:, left_index * 3 : left_index * 3 + 3, left_index * 3 : left_index * 3 + 3] = matrix

            Sigma_R_i = torch.zeros(self.N_E_points, self.hessian.size(-1), self.hessian.size(-1), dtype=data_type_complex, device=device)
            right_index = self.right_indices[i]
            matrix = torch.eye(3, dtype=data_type_complex, device=device) * self.sigma[:, None, None]
            Sigma_R_i[:, right_index * 3: right_index * 3 + 3, right_index * 3: right_index * 3 + 3] = matrix

            Gamma_L_i = (-2. * torch.imag(Sigma_L_i)).to(data_type_complex) # Shape (N_E, N_orb, N_orb)
            Gamma_R_i = (-2. * torch.imag(Sigma_R_i)).to(data_type_complex) # Shape (N_E, N_orb, N_orb)

            #correct hessian -> momentum conservation
            hessian_i = self.hessian[i]

            gamma_hb = self.gamma * __eV2Ha__ / __ang2bohr__ ** 2
            matrix = torch.eye(3, dtype=data_type_real, device=device) * gamma_hb
            hessian_i[right_index * 3: right_index * 3 + 3, right_index * 3: right_index * 3 + 3] -= matrix/self.M_C
            hessian_i[left_index * 3: left_index * 3 + 3, left_index * 3: left_index * 3 + 3] -= matrix/self.M_C


            #mass weighted hessian
            masses = self.el_structure_calculator.numbers[i].repeat_interleave(3).cpu().numpy()
            masses = torch.tensor(np.array([ATOM_DICT_ANUM[m][2] if m != 0 else 1.0 for m in masses]), device=device, dtype=data_type_real)
            masses = torch.sqrt(torch.outer(masses, masses))
            hessian_i = hessian_i / masses

            eta = 1E-9 * 0
            w_sq = (self.w * (1 + 1.j * eta)) ** 2
            w_sq = w_sq.view(-1, 1, 1)
            identity = torch.eye(hessian_i.size(-1), dtype=data_type_complex, device=device)

            hessian_i = hessian_i.unsqueeze(0).to(data_type_complex)
            matrix_to_invert = w_sq*identity - hessian_i - Sigma_L_i - Sigma_R_i

            #handle zero padded stuff
            matrix_to_invert = torch.where(block_mask[i], matrix_to_invert, identity_batch[i])
            G_r_CC_i = torch.linalg.solve(matrix_to_invert, identity_batch[i])

            # Compute inverse
            G_a_CC_i = torch.conj(G_r_CC_i.transpose(-2, -1)) # Shape (N_E, N_orb, N_orb)

            del Sigma_L_i
            del Sigma_R_i

            # Compute transmission tau for the current batch element across all energies
            Gamma_L_i = Gamma_L_i.expand(self.w.size(0), -1, -1)
            Gamma_R_i = Gamma_R_i.expand(self.w.size(0), -1, -1)

            #mat_i = torch.bmm(torch.bmm(Gamma_L_i, G_r_CC_i), torch.bmm(Gamma_R_i, G_a_CC_i)) # Shape (N_E, N_orb, N_orb)
            mat_i = torch.einsum('bij,bjk,bkl,blm->bim', Gamma_L_i, G_r_CC_i, Gamma_R_i, G_a_CC_i)

            # Trace and take the real part
            trace_i = torch.diagonal(mat_i, dim1=-2, dim2=-1).sum(-1) # Shape (N_E,)
            tau_i = torch.real(trace_i) # Shape (N_E,)

            # Store the result for the current batch element
            tau_all_batches[i] = tau_i


        return tau_all_batches


    @torch.no_grad()
    def calculate_tau_ph(self):
        """
        Calculate the phonon transmission using NEGF. Batched over the molecules in the el_structure_calculator.
        """


        device = self.dd["device"]
        dtype_real = self.w.dtype
        dtype_complex = torch.complex64 if dtype_real == torch.float32 else torch.complex128

        # Constants
        gamma_hb = self.gamma * __eV2Ha__ / __ang2bohr__ ** 2
        eta = 1E-9
        gamma_val = -2. * torch.imag(self.sigma)  # Pre-calculate once

        # Matrix correction
        matrix_correction = torch.eye(3, dtype=dtype_real, device=device) * gamma_hb / self.M_C

        tau_all_batches = torch.empty(self.batch_size, self.w.shape[0], device=device, dtype=dtype_real)

        # Zero-padding masks
        diagonal_mask = torch.diagonal(self.hessian, dim1=-2, dim2=-1) != 0
        valid_sizes = torch.sum(diagonal_mask, dim=1)
        rows_and_cols = torch.arange(self.hessian[0].size(-1), device=device)
        block_mask = ((rows_and_cols[None, :, None] < valid_sizes[:, None, None]) &
                      (rows_and_cols[None, None, :] < valid_sizes[:, None, None]))


        hessian_size = self.hessian[0].size(-1)
        identity_matrix = torch.eye(hessian_size, dtype=dtype_complex, device=device)

        w_sq = (self.w * (1 + 1.j * eta)) ** 2
        sigma_expanded = self.sigma.view(-1, 1).expand(-1, 3).flatten()

        for i in range(self.batch_size):

            left_index = self.left_indices[i]
            right_index = self.right_indices[i]
            left_slice = slice(left_index * 3, left_index * 3 + 3)
            right_slice = slice(right_index * 3, right_index * 3 + 3)

            hessian_i = self.hessian[i].clone()

            masses = self.el_structure_calculator.numbers[i].repeat_interleave(3).cpu().numpy()
            masses = torch.tensor([ATOM_DICT_ANUM[m][2] if m != 0 else 1.0 for m in masses],
                                  device=device, dtype=dtype_real)
            masses = torch.sqrt(torch.outer(masses, masses))
            hessian_i.div_(masses)

            hessian_i[right_slice, right_slice] -= matrix_correction
            hessian_i[left_slice, left_slice] -= matrix_correction

            hessian_i_complex = hessian_i.to(dtype_complex)
            #del hessian_i, masses

            #sigma corrections
            diagonal_part = torch.diag_embed(w_sq.unsqueeze(1).expand(-1, hessian_size))
            diagonal_part.diagonal(dim1=-2, dim2=-1)[:, left_slice] -= sigma_expanded.view(-1, 3)
            diagonal_part.diagonal(dim1=-2, dim2=-1)[:, right_slice] -= sigma_expanded.view(-1, 3)

            diagonal_part -= hessian_i_complex

            # Zero-padding mask
            identity_expanded = identity_matrix.expand(self.w.shape[0], -1, -1)
            diagonal_part = torch.where(block_mask[i], diagonal_part, identity_expanded)

            try:
                G_r_CC_i = torch.linalg.solve(diagonal_part, identity_expanded)
                G_r_LR = G_r_CC_i[:, left_slice, right_slice]
                # Calculate transmission
                tau_i = gamma_val ** 2 * torch.sum(torch.abs(G_r_LR) ** 2, dim=(-1, -2))
                tau_all_batches[i] = tau_i.real
                del G_r_CC_i, G_r_LR

            except torch.linalg.LinAlgError:
                tau_all_batches[i] = 0.0
                print(f"Warning: Singular matrix encountered in batch {i}")

            del hessian_i_complex, diagonal_part

        del identity_matrix, block_mask, diagonal_mask
        torch.cuda.empty_cache()

        return tau_all_batches


    @torch.no_grad()
    def calculate_kappa_ph(self):


        #convert to Har
        E = self.E * 1E-3 * __eV2Ha__
        prefactor = 1.0 / (__h__ * __k_B__ * self.T ** 2)
        beta = 1.0 / (__k_B__ * self.T)

        exp_ = torch.exp(E * beta)
        exp_ = torch.where(torch.isinf(exp_), 0.0, exp_)
        integrand = E ** 2 * self.tau_ph * exp_ / ((exp_ - 1) ** 2)
        integrand = torch.nan_to_num(integrand, nan=0.0)
        integral = torch.trapz(integrand, E, dim=-1)
        #todo: fix units
        kappa = prefactor * integral * __har2pJ__

        return kappa




if __name__ == '__main__':
    pass





