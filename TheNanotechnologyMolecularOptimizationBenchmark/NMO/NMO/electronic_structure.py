import tad_mctc as mctc
import torch
import dxtb
import numpy as np
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import time
from collections import deque
import re

from tad_mctc.units import AA2AU
from tad_mctc.io.write import write_xyz
import logging
from ase import Atoms
from ase.optimize import BFGS
from ase.units import Hartree, Bohr  # For unit conversions
from rdkit import Chem
from rdkit.Chem import AllChem
import tempfile
import os
import subprocess


from .utils import print_vram_usage, read_xyz_file, ANG2BOHR, ATOM_DICT_SYM, read_hessian, read_coord_file, ATOM_DICT_ANUM, write_xyz_file

__dtype__ = torch.float64
#timeout per xtb process of 25 minutes
TIMEOUT = 25 * 60
HL_GAP_THRESH = 0.2


def _run_single_terahertz_external(args):
    """
    Worker function for geometry optimization.
    Args:
        args: tuple(numbers, positions, i, dtype_str, n_threads, gfn)
    """
    numbers, positions, i, gfn, n_threads = args


    ram_dir = "/dev/shm" if os.path.exists("/dev/shm") else None
    value = -1.0
    try:
        with tempfile.TemporaryDirectory(dir=ram_dir) as tmpdir:

            t_nums = torch.tensor(numbers)
            t_pos = torch.tensor(positions)
            xyz_file = os.path.join(tmpdir, "mol.xyz")
            write_xyz(xyz_file, t_nums, t_pos, overwrite=True)
            xtb_ptb_bin = os.environ.get("XTB_PTB_BIN")
            if not xtb_ptb_bin:
                raise RuntimeError(
                    "Environment variable XTB_PTB_BIN is not set. "
                    "It must point to the xtb binary that supports the calculation of terahertz upconversion (see README)."
                )
            cmd = [xtb_ptb_bin, '--gfn', str(gfn), '--ptb', '--hess', '--raman', '289.15', '785', 'mol.xyz', '-P', str(n_threads)]

            # DYNAMIC THREADING
            my_env = os.environ.copy()
            my_env["OMP_NUM_THREADS"] = str(n_threads)
            my_env["MKL_NUM_THREADS"] = str(n_threads)
            my_env["OPENBLAS_NUM_THREADS"] = str(n_threads)
            my_env["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
            my_env["NUMEXPR_NUM_THREADS"] = str(n_threads)
            my_env["BLIS_NUM_THREADS"] = str(n_threads)

            result = subprocess.run(
                cmd,
                cwd=tmpdir,
                check=True,
                capture_output=True,
                text=True,
                env=my_env,
                timeout = TIMEOUT
            )

            #extract terahertz frequencies from stdout
            pattern = r"THz target P:\s+(\S+)"
            match = re.search(pattern, result.stdout)


            if match:
                value_str = match.group(1)
                try:
                    value = float(value_str)
                    if np.isnan(value):
                        print(f"Result is invalid (NaN). Raw string: {value_str}")
                        converged = False
                    else:
                        print(f"Extracted Value: {value}")
                        converged = True

                except ValueError:
                    print(f"Could not convert '{value_str}' to a number.")
                    converged = False
            else:
                print("Pattern not found in output.")
                converged = False

    except Exception as e:
        #logging.debug(f"Mol {i}: terahertz calculation failed {e}")
        converged = False

    value = 10**(value)

    return i, converged, value

def _run_single_opt_external(args):
    """
    Worker function for geometry optimization.
    Args:
        args: tuple(numbers, positions, i, dtype_str, n_threads, gfn)
    """
    numbers, positions, i, target_dtype_str, gfn, n_threads = args

    # Reconstruct correct dtype
    target_dtype = torch.float64 if target_dtype_str == "float64" else torch.float32

    # Defaults
    res_numbers = None
    res_positions = None
    converged = False
    hl_gap = -1
    failure_reason = ""

    ram_dir = "/dev/shm" if os.path.exists("/dev/shm") else None

    try:
        with tempfile.TemporaryDirectory(dir=ram_dir) as tmpdir:
            # Write XYZ (using manual writing to avoid torch overhead in subprocess if desired,
            # but here we use mctc for consistency)
            t_nums = torch.tensor(numbers)
            t_pos = torch.tensor(positions)
            xyz_file = os.path.join(tmpdir, "mol.xyz")
            write_xyz(xyz_file, t_nums, t_pos, overwrite=True)

            cmd = ['xtb', '--opt', 'extreme', '--gfn', str(gfn), 'mol.xyz', '-P', str(n_threads)]

            # DYNAMIC THREADING
            my_env = os.environ.copy()
            my_env["OMP_NUM_THREADS"] = str(n_threads)
            my_env["MKL_NUM_THREADS"] = str(n_threads)
            my_env["OPENBLAS_NUM_THREADS"] = str(n_threads)
            my_env["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
            my_env["NUMEXPR_NUM_THREADS"] = str(n_threads)
            my_env["BLIS_NUM_THREADS"] = str(n_threads)

            result = subprocess.run(
                cmd,
                cwd=tmpdir,
                check=True,
                capture_output=True,
                text=True,
                env=my_env,
                timeout=TIMEOUT
            )

            output = result.stdout
            gap_pattern = r"HOMO-LUMO\s+GAP\s+(?:\|)?\s+([\d\.]+)\s+eV"
            gap_matches = re.findall(gap_pattern, output, re.IGNORECASE)

            if gap_matches:
                hl_gap = float(gap_matches[-1])

            if hl_gap < HL_GAP_THRESH:
                raise ValueError(f"HOMO-LUMO gap too small {hl_gap=}")

            xyz_path = os.path.join(tmpdir, "xtbopt.xyz")
            ok_file = os.path.join(tmpdir, ".xtboptok")
            topo_file = os.path.join(tmpdir, "xtbtopo.mol")
            if os.path.exists(xyz_path) and os.path.exists(ok_file) and os.path.exists(topo_file):
                coord_xyz = read_xyz_file(xyz_path)
                atom_types = coord_xyz[0, :]
                positions_T = coord_xyz[1:4, :].T

                positions_np = np.array(positions_T, dtype=float) * ANG2BOHR
                numbers_np = np.array([float(ATOM_DICT_SYM[item.lower()][0]) for item in atom_types], dtype=int)

                res_numbers = numbers_np
                res_positions = positions_np
                converged = True

                # check topology
                mol = Chem.MolFromMolFile(topo_file, sanitize=False)
                if mol is None:
                    raise ValueError("Optimized mol file could not be read, optimization likely failed.")
                mol.UpdatePropertyCache(strict=False)
                fragments = Chem.GetMolFrags(mol)
                num_fragments = len(fragments)
                if num_fragments > 1:
                    raise ValueError(f"Optimization resulted in multiple fragments: {num_fragments}")

                #check if gold (if present) has more than one bond -> not reasonable
                gold_issues = []
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() == 'Au':
                        bond_count = atom.GetDegree()
                        if bond_count > 1:
                            gold_issues.append((atom.GetIdx(), bond_count))
                if gold_issues:
                    raise ValueError("Gold is connected to multiple atoms")

            else:
                failure_reason = "xtb opt output files missing (xtbopt.xyz / .xtboptok / xtbtopo.mol not found)"

    except Exception as e:
        #logging.debug(f"Mol {i}: Optimization failed {e}")
        converged = False
        failure_reason = str(e)

    # Fallback for failures
    if not converged:
        res_numbers = np.zeros(3, dtype=int)
        res_positions = np.zeros((3, 3), dtype=float)

    return i, converged, res_numbers, res_positions, hl_gap, failure_reason


def _run_single_hessian_external(args):
    """
    Worker function for external hessian calculation. This is implemented because dxtb does not support stable calculation
    of hessians yet.
    Args:
        args: tuple(numbers, positions, i, unpadded_length, n_threads, gfn)
    """
    numbers, positions, i, unpadded_length, gfn, n_threads = args

    hessian_res = None
    valid = False
    failure_reason = ""
    ram_dir = "/dev/shm" if os.path.exists("/dev/shm") else None

    try:
        with tempfile.TemporaryDirectory(dir=ram_dir) as tmpdir:
            t_nums = torch.tensor(numbers)
            t_pos = torch.tensor(positions)
            xyz_file = os.path.join(tmpdir, "mol.xyz")
            write_xyz(xyz_file, t_nums, t_pos, overwrite=True)

            cmd = ['xtb', '--hess', '--gfn', str(gfn), 'mol.xyz', '-P', str(n_threads)]

            # DYNAMIC THREADING
            my_env = os.environ.copy()
            my_env["OMP_NUM_THREADS"] = str(n_threads)
            my_env["MKL_NUM_THREADS"] = str(n_threads)
            my_env["OPENBLAS_NUM_THREADS"] = str(n_threads)
            my_env["VECLIB_MAXIMUM_THREADS"] = str(n_threads)
            my_env["NUMEXPR_NUM_THREADS"] = str(n_threads)
            my_env["BLIS_NUM_THREADS"] = str(n_threads)

            subprocess.run(
                cmd,
                cwd=tmpdir,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=my_env,
                timeout=TIMEOUT
            )

            hess_path = os.path.join(tmpdir, "hessian")
            not_ok_path = os.path.join(tmpdir, "xtbhess.xyz")
            if os.path.exists(hess_path) and not os.path.exists(not_ok_path):
                hessian_res = read_hessian(hess_path, unpadded_length)
                valid = True
            else:
                failure_reason = "xtb hessian output file missing or xtbhess.xyz present (xtb did not converge)"

    except Exception as e:
        #logging.debug(f"Mol {i}: Hessian calc failed {e}")
        valid = False
        failure_reason = str(e)

    if not valid:
        hessian_res = np.zeros((unpadded_length * 3, unpadded_length * 3))

    return i, valid, hessian_res, failure_reason




class Electronic_Structure_Calculator():
    """
    A class to calculate electronic structure properties using the dxtb calculator.
    """

    def __init__(self, numbers, positions, charge = None, fixed_atoms = None, opts_add = None, add_fake_batch = True, gfn = 1):
        """
        Initialize the Electronic_Structure_Calculator.
        Args:
            numbers (torch.Tensor): Atomic numbers of the atoms.
            positions (torch.Tensor): Positions of the atoms in Bohr.
            charge (torch.Tensor): Charge of the system.
            fixed_atoms (torch.Tensor): Mask for fixed
            opts_add (dict): Additional options for the dxtb calculator.
            add_fake_batch (bool): Whether to add a fake batch dimension if input is a single molecule.
            gfn (str): xtb parametrization
        """

        self.numbers = numbers
        self.positions = positions

        self.gfn = gfn


        self.valid_batch_indices = np.arange(self.numbers.shape[0])

        if len(self.numbers.shape) == 1 and add_fake_batch:
            #if positions is 2D, we assume it is a single molecule
            print("Fake batch")
            self.positions = self.positions.unsqueeze(0)
            self.numbers = self.numbers.unsqueeze(0)
            pass

        if torch.cuda.is_available():
            #somehow tad_mctc does not like just "cuda"
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.fixed_atoms = fixed_atoms

        dd = {"dtype": __dtype__, "device": torch.device("cpu")}
        if torch.cuda.is_available():
            dd["device"] = self.device
        #"scf_mode": dxtb.labels.SCF_MODE_FULL,
        #SCP_MODE_CHARGE_STRS
        opts = {
            "cache_enabled": True,
            "cache_charges": True,
            "cache_coefficients": True,
            "cache_mo_energies": True,
            "cache_overlap": True,
            "cache_occupation": True,
            "damp": 0.1, #to high damping can lead to problems -> Fermi energy is not converging
            "damp_dynamic" : True, #this is new
            "scf_mode": dxtb.labels.SCF_MODE_FULL,
            "scp_mode": dxtb.labels.SCP_MODE_CHARGE,
            "verbosity": 0,
            "fermi_maxiter" : 200,
            "fermi_etemp" : 300
        }

        if opts_add is not None:
            for key, value in opts_add.items():
                opts[key] = value
        if self.gfn == 1:
            self.calculator = dxtb.calculators.GFN1Calculator(self.numbers, opts=opts, **dd)
        elif self.gfn == 2:
            self.calculator = dxtb.calculators.GFN2Calculator(self.numbers, opts=opts, **dd)
        else:
            raise ValueError(f"gfn parametrization {self.gfn} not supported")

        self._energy = None
        self._coefficients = None
        self._mo_energies = None
        self._overlap = None
        self._occupation = None
        self._charge = None
        self._hessian = None

    def __call__(self, positions_ang):
        """
        Provides the energy and gradients for given positions in Angstrom.
        Args:
            positions_ang (torch.Tensor): Positions in Angstrom. Note that this has different units than the internal positions (Bohr).
        """
        if not positions_ang.requires_grad:
            positions_ang.requires_grad = True

        if positions_ang.grad is not None:
            positions_ang.grad.zero_()

        positions_ang.data = positions_ang.data * ANG2BOHR
        self.positions = positions_ang

        # The calculator expects a flat (N_atoms_total, 3) tensor
        energy_hartree = self.calculator.get_energy(self.positions, self.charge)
        energy_hartree_sum = energy_hartree.sum()
        energy_hartree_sum.backward()

        gradients_hartree_per_bohr = self.positions.grad

        if self.fixed_atoms is not None:
            with torch.no_grad():
                fixed_atoms_mask = self.fixed_atoms.unsqueeze(-1)
                gradients_hartree_per_bohr.data.mul_(fixed_atoms_mask)

        return energy_hartree.detach().cpu().numpy(), gradients_hartree_per_bohr.detach().cpu().numpy()


    @classmethod
    def from_coord_xyz(cls, coord_xyz_path, opts_add = None, gfn = 1, dd = {"dtype": __dtype__, "device": torch.device("cpu")}):
        """
        Create an instance of the class from a coord_xyz file.

        Args:
            coord_xyz (str): Path to the coordinate file.
            opts_add (dict): Options for the dxtb calculator.
            gfn (str): xtb parametrization

        Returns:
            electronic_structure_calculator: An instance of the class.
        """

        # Read the coordinates from the file
        coord_xyz = read_xyz_file(coord_xyz_path)

        # Extract numbers and positions
        numbers = coord_xyz[0,:]
        numbers = [ATOM_DICT_SYM[n.lower()][0] for n in numbers]
        numbers = torch.tensor(numbers, device=dd["device"])
        positions = np.array(coord_xyz[1:4].T, dtype=float)
        #convert to Bohr
        positions = positions * AA2AU
        positions = torch.tensor(positions, **dd)

        return cls(numbers, positions, opts_add = opts_add, gfn = gfn)

    @classmethod
    def from_coord_xyz_batch(cls, coord_xyz_path_list, opts_add = None, gfn = 1,  dd = {"dtype": __dtype__, "device": torch.device("cpu")}):
        if torch.cuda.is_available():
            dd["device"] = torch.device("cuda")
        numbers_tensor_list = []
        positions_tensor_list = []
        for path in coord_xyz_path_list:
            coord_xyz = read_xyz_file(path)
            # Extract numbers and positions
            numbers = coord_xyz[0, :]
            valid_numbers = np.array([i for i, n in enumerate(numbers) if n != "X"])
            numbers = [ATOM_DICT_SYM[n.lower()][0] for n in numbers[valid_numbers]]
            numbers = torch.tensor(numbers, device=dd["device"])
            positions = np.array(coord_xyz[1:4, valid_numbers].T, dtype=float)
            # convert to Bohr
            positions = positions * AA2AU

            positions = torch.tensor(positions, **dd)
            numbers_tensor_list.append(numbers)
            positions_tensor_list.append(positions)

        if len(coord_xyz_path_list) > 1:
            positions = mctc.batch.pack(positions_tensor_list)
            numbers = mctc.batch.pack(numbers_tensor_list)
        else:
            positions = positions_tensor_list[0]
            numbers = numbers_tensor_list[0]

        return cls(numbers, positions, opts_add = opts_add, gfn = gfn)

    @classmethod
    def from_coord_batch(cls, coord_file_list, opts_add = None, gfn = "gfn1", dd = {"dtype": __dtype__, "device": torch.device("cpu")}):
        """
        Create an instance of the class from a list of coordinate files. This also reads fixed atoms.
        Args:
            coord_file_list (list): List of paths to coordinate files.
            opts_add (dict): Options for the dxtb calculator.
            gfn (str): xtb parametrization
            dd (dict): Data type and device settings.
        """
        if torch.cuda.is_available():
            dd["device"] = torch.device("cuda")
        numbers_tensor_list = []
        positions_tensor_list = []
        fixed_atoms_tensor_list = []
        for path in coord_file_list:
            coord_xyz = read_coord_file(path)
            # Extract numbers and positions
            numbers = coord_xyz[3, :]
            numbers = [ATOM_DICT_SYM[n.lower()][0] for n in numbers]
            numbers = torch.tensor(numbers, device=dd["device"])
            positions = np.array(coord_xyz[0:3].T, dtype=float)
            positions = torch.tensor(positions, **dd)

            fixed_atoms = coord_xyz[4, :]
            fixed_atoms = [0 if symbol == 'f' else 1 for symbol in fixed_atoms]
            fixed_atoms = torch.tensor(fixed_atoms, device=dd["device"])


            numbers_tensor_list.append(numbers)
            positions_tensor_list.append(positions)
            fixed_atoms_tensor_list.append(fixed_atoms)

        positions = mctc.batch.pack(positions_tensor_list)
        numbers = mctc.batch.pack(numbers_tensor_list)
        fixed_atoms = mctc.batch.pack(fixed_atoms_tensor_list)

        return cls(numbers, positions, fixed_atoms =  fixed_atoms, opts_add = opts_add, gfn = gfn)

    @classmethod
    def from_mol_batch(cls, mol_list, opts_add = None, gfn = 1, dd = {"dtype": __dtype__, "device": torch.device("cpu")}):
        """
        Create an instance of the class from a list of rdkit mol objects.
        Args:
            mol_list (list): List of rdkit mol objects.
            opts_add (dict): Options for the dxtb calculator.
            gfn (str): xtb parametrization
            dd (dict): Data type and device settings.
        """
        if torch.cuda.is_available():
            dd["device"] = torch.device("cuda")
        numbers_tensor_list = []
        positions_tensor_list = []


        for mol in mol_list:
            numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            numbers = torch.tensor(numbers, device=dd["device"])
            if mol.GetNumConformers() == 0:
                id = Chem.AllChem.EmbedMolecule(mol, randomSeed=0)
                if id == -1:
                    raise ValueError("Could not embed molecule")
            conf = mol.GetConformer()
            positions = [conf.GetAtomPosition(atom.GetIdx()) for atom in mol.GetAtoms()]
            positions = torch.tensor(positions, **dd)

            # convert to Bohr
            positions = positions * AA2AU

            numbers_tensor_list.append(numbers)
            positions_tensor_list.append(positions)


        positions = mctc.batch.pack(positions_tensor_list)
        numbers = mctc.batch.pack(numbers_tensor_list)
        return cls(numbers, positions, opts_add = opts_add, gfn = gfn)

    @property
    def charge(self):
        """
        Get the charge of the system. If not set, it initializes to zero. -> takes care of batching.
        Returns:
            torch.Tensor: The charge of the system.
        """
        if self._charge is None:
            #batching
            if len(self.positions.shape) == 3 and self.positions.shape[0] > 1:
                self._charge = torch.tensor([0.0]*self.positions.shape[0], dtype=__dtype__, device=self.positions.device)
            else:
                self._charge = torch.tensor([0.0], dtype=__dtype__, device=self.positions.device)
                self._charge = self._charge.unsqueeze(0)
        return self._charge

    @property
    def energy(self):
        """
        Calculate the energy of the system.

        Returns:
            float: The calculated energy.
        """
        if self._energy is None:
            self._energy = self.calculator.get_energy(self.positions, self.charge)
        return self._energy

    @property
    def coefficients(self):
        """
        Calculate the coefficients of the system.

        Returns:
            np.ndarray: The calculated coefficients.
        """
        if self._coefficients is None:
            self._coefficients = self.calculator.get_coefficients(self.positions, self.charge)
        return self._coefficients

    @property
    def mo_energies(self):
        """
        Calculate the molecular orbital energies of the system.

        Returns:
            np.ndarray: The calculated molecular orbital energies.
        """
        if self._mo_energies is None:
            self._mo_energies = self.calculator.get_mo_energies(self.positions, self.charge)
        return self._mo_energies

    @property
    def overlap(self):
        """
        Calculate the overlap matrix of the system.

        Returns:
            np.ndarray: The calculated overlap matrix.
        """
        if self._overlap is None:
            self._overlap = self.calculator.integrals.overlap.matrix
        return self._overlap

    @property
    def occupation(self):
        """
        Calculate the occupation numbers of the molecular orbitals.

        Returns:
            np.ndarray: The calculated occupation numbers.
        """
        if self._occupation is None:
            self._occupation = self.calculator.get_occupation(self.positions, self.charge)
        return self._occupation

    @property
    def hessian(self):
        """
        Calculate the Hessian matrix of the system.

        Returns:
            np.ndarray: The calculated Hessian matrix.
        """
        if self._hessian is None:
            self._hessian = self.get_hessian_external_xtb()
        return self._hessian

    def _get_optimal_threads(self, n_atoms, max_cpus_limit):
        """
        Determines the optimal number of threads based on system size and scaling.
        Boundaries are chosen based on O(N^4) hessian scaling and O(N^3) scf scaling.

        Classes:
          - Very Small (<= 15 atoms): 1 CPU
          - Small (16-30 atoms): 2 CPUs
          - Medium (31-55 atoms): 4 CPUs
          - Large (> 55 atoms): 8 CPUs
        """
        if n_atoms <= 15:
            return 1
        elif n_atoms <= 30:
            return min(2, max_cpus_limit)
        elif n_atoms <= 55:
            return min(4, max_cpus_limit)
        else:
            return min(8, max_cpus_limit)

    def get_hessian_dxtb(self):
        """
        Calculates hessian. Creates new calculators for every batch item because dxtb does not support batching for hessian calculation yet.
        """

        opts = {
            "cache_enabled": False,
            "cache_charges": False,
            "cache_coefficients": False,
            "cache_mo_energies": False,
            "cache_overlap": False,
            "cache_occupation": False,
            "damp": 0.1,  # to high damping can lead to problems -> Fermi energy is not converging
            "damp_dynamic": True,  # this is new
            "scf_mode": dxtb.labels.SCF_MODE_FULL,
            "scp_mode": dxtb.labels.SCP_MODE_CHARGE,
            "verbosity": 0,
            "fermi_maxiter": 200,
            "fermi_etemp": 300
        }


        hessians = []
        valid = np.ones(self.positions.shape[0], dtype=bool)
        #logging.debug(f"valid: {valid}")
        for i in range(self.positions.shape[0]):
            try:
                #print_vram_usage()
                #print(f"Calculating hessian for molecule {i}")
                # Filter out padding atoms
                active_mask = (self.numbers[i] != 0)
                active_numbers = self.numbers[i][active_mask]
                unpadded_length = len(active_numbers)
                numbers = self.numbers[i, :unpadded_length]
                if self.gfn == 1:
                    calculator_tmp = dxtb.calculators.GFN1Calculator(numbers, opts=opts, dtype=__dtype__, device=self.positions.device)
                elif self.gfn == 2:
                    calculator_tmp = dxtb.calculators.GFN2Calculator(numbers, opts=opts, dtype=__dtype__, device=self.positions.device)
                else:
                    raise ValueError(f"Unknown GFN: {self.gfn}")
                positions = self.positions[i, :unpadded_length, :].detach().clone()
                positions.requires_grad_(True)
                hessian_i = calculator_tmp.get_hessian(positions, 0, derived_quantity = "energy", use_functorch = False)
                hessian_i = hessian_i.reshape(hessian_i.shape[0] * 3, hessian_i.shape[0] * 3)

            except torch.cuda.OutOfMemoryError:
                print(f"OOM Error for molecule {i} during hessian calculation and N_atoms = {unpadded_length}")
                #dummy hessian
                hessian_i = torch.zeros((unpadded_length * 3, unpadded_length * 3), dtype=__dtype__, device=self.positions.device)
                valid[i] = False


            hessian_ = hessian_i.detach().clone()
            hessians.append(hessian_)
            del positions
            del hessian_i
            del calculator_tmp
            torch.cuda.empty_cache()

        #logging.debug(f"valid end of hessian: {valid}")
        self.valid_batch_indices = self.valid_batch_indices[valid]
        #logging.debug(f"self.valid_batch_indices: {self.valid_batch_indices}")

        hessians = mctc.batch.pack(hessians)
        hessians.requires_grad_(False)
        torch.cuda.empty_cache()
        #print_vram_usage()
        return hessians

    def _dynamic_submitter(self, tasks, worker_func, total_cpus):
        """
        Submit tasks dynamically with CPU-aware scheduling using threads.
        """
        results = []
        running_futures = {}  # {future: (cpu_cost, original_index)}
        available_cpus = total_cpus

        task_queue = deque(enumerate(tasks))  # Track original indices
        max_thread_workers = max(total_cpus * 2, 4)

        with ThreadPoolExecutor(max_workers=max_thread_workers) as executor:
            while task_queue or running_futures:

                # Submit new tasks while we have capacity
                while task_queue:
                    idx, task_args = task_queue[0]
                    cpu_cost = task_args[-1]

                    if available_cpus >= cpu_cost:
                        task_queue.popleft()
                        future = executor.submit(worker_func, task_args)
                        running_futures[future] = (cpu_cost, idx)
                        available_cpus -= cpu_cost
                    else:
                        break  # Wait for resources to free up

                # Wait for at least one task to complete (or timeout)
                if running_futures:
                    done, _ = wait(running_futures.keys(), timeout=0.1,
                                   return_when=FIRST_COMPLETED)

                    for future in done:
                        cpu_cost, idx = running_futures.pop(future)
                        available_cpus += cpu_cost

                        try:
                            result = future.result()
                            results.append((idx, result))  # Preserve order
                        except Exception as e:
                            #logging.error(f"Task {idx} failed: {e}")
                            print("Task {idx} failed: {e}")
                            results.append((idx, None))  # Or handle differently

        # Sort by original index if order matters
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def get_hessian_external_xtb(self, total_cpus=None):
        """
        Parallel Hessian calculation with dynamic CPU allocation. This uses external xtb calls as dxtb does not support stable hessian calculation yet.
        """
        if total_cpus is None:
            total_cpus = os.cpu_count() or 4

        #logging.debug(f"Starting dynamic hessian with {total_cpus} CPUs")

        tasks = []
        unpadded_lengths = {}

        for i in range(self.positions.shape[0]):
            active_mask = (self.numbers[i] != 0)
            n_atoms = int(active_mask.sum().item())
            unpadded_lengths[i] = n_atoms

            nums_np = self.numbers[i, :n_atoms].detach().cpu().numpy()
            pos_np = self.positions[i, :n_atoms, :].detach().cpu().numpy()

            # Determine Threads
            n_threads = self._get_optimal_threads(n_atoms, total_cpus)

            # Pack args: (nums, pos, i, unpadded_length, n_threads)
            tasks.append((nums_np, pos_np, i, n_atoms, self.gfn, n_threads))

        # Run Scheduler
        results = self._dynamic_submitter(tasks, _run_single_hessian_external, total_cpus)

        # Sort and Pack
        results.sort(key=lambda x: x[0])

        hessian_tensors = []
        valid_indices = []
        hess_failure_reasons = [""] * self.positions.shape[0]

        for idx, valid, hess_np, failure_reason in results:
            if valid:
                valid_indices.append(idx)
            else:
                hess_failure_reasons[idx] = failure_reason

            h_tens = torch.tensor(hess_np, dtype=__dtype__, device=self.device)
            hessian_tensors.append(h_tens)

        # Update valid indices mask
        current_valid_mask = np.zeros(self.positions.shape[0], dtype=bool)
        current_valid_mask[valid_indices] = True
        self.valid_batch_indices = self.valid_batch_indices[current_valid_mask]

        hessians = mctc.batch.pack(hessian_tensors)
        hessians.requires_grad_(False)

        return hessians, hess_failure_reasons



    def reset_cache(self):
        """
        Reset the cache of the calculator.
        """
        self._energy = None
        self._coefficients = None
        self._mo_energies = None
        self._overlap = None
        self._occupation = None

    def turn_off_gradients(self):
        """
        Turn off gradients for the positions tensor.
        """
        self.positions.requires_grad_(False)


    def optimize_geometry(self,
            steps: int = 1000,
            gradient_tolerance: float = 1e-3,
            fmax_tolerance: float = 1e-3,  # Typically check max force component
    ) -> torch.Tensor:
        """
        Optimizes the geometry using dxtb and ase optimizers.

        Args:
            steps (int): Maximum number of optimization steps.
            gradient_tolerance (float):
            fmax_tolerance (float):

        Returns:
            converged (list): Boolean tensor indicating convergence for each molecule in the batch.
        """

        atoms_list = []
        optimizers = []
        unpadded_length = [0] * self.positions.shape[0]

        for i in range(self.positions.shape[0]):
            numbers = self.numbers[i].detach().cpu().numpy()

            # Filter out padding atoms
            active_mask = (numbers != 0)
            active_numbers = numbers[active_mask]
            unpadded_length[i] = len(active_numbers)

            # ASE uses angstrom
            initial_coords = self.positions[i, :unpadded_length[i], :].detach().cpu().numpy() / ANG2BOHR

            mol = Atoms(numbers=active_numbers, positions=initial_coords)
            atoms_list.append(mol)
            optimizers.append(BFGS(mol))

        converged = [False] * self.positions.shape[0]
        step = 0
        history = []


        HARTREE_PER_BOHR_TO_EV_PER_ANG = Hartree / Bohr

        while not all(converged):
            #print("vram geo_opt")
            #print_vram_usage()
            coords_list = []
            for i, is_conv in enumerate(converged):
                if not is_conv:
                    coords = atoms_list[i].get_positions()
                else:
                    # If converged, use its last coordinates
                    coords = atoms_list[i].get_positions()
                coords_list.append(torch.tensor(coords, dtype=torch.float64).to(self.positions.device))

            history.append(coords_list)
            coords_batch_tensor = mctc.batch.pack(coords_list)

            energies_Ha, gradients_Ha_Bohr = self(coords_batch_tensor)

            print(f"Step {step}")
            for i in range(self.positions.shape[0]):
                if not converged[i]:
                    forces_eV_Ang = -gradients_Ha_Bohr[i, :unpadded_length[i], :] * HARTREE_PER_BOHR_TO_EV_PER_ANG

                    optimizers[i].step(forces_eV_Ang)

                    energy_eV = energies_Ha[i].item() * Hartree
                    max_force = np.sqrt((forces_eV_Ang ** 2).sum(axis=1).max())
                    max_force_ha_bohr  = max_force / HARTREE_PER_BOHR_TO_EV_PER_ANG
                    gradient_norm = np.linalg.norm(gradients_Ha_Bohr[i, :unpadded_length[i], :])
                    print(f"  Mol {i + 1}: energy = {energy_eV:12.8f} eV, Max Force = {max_force:12.8f} eV/A = {max_force_ha_bohr:12.8f} Ha/Bohr, grad_norm = {gradient_norm:12.8f}")


                    if max_force_ha_bohr < fmax_tolerance and gradient_norm < gradient_tolerance:
                        #print(f"  Mol {i + 1}: Converged!")
                        converged[i] = True

            if step > steps:
                print("Optimization stopped: max steps reached.")
                break
            step += 1

        final_coords_list = [torch.tensor(atoms.get_positions(), dtype=torch.float64).to(self.positions.device) for
                             atoms in atoms_list]
        self.positions = mctc.batch.pack(final_coords_list)
        self.positions.data = self.positions.data * ANG2BOHR
        return converged

    def optimize_geometry_external(self, total_cpus=None):
        """
        Parallel geometry optimization with dynamic CPU allocation.
        """
        if total_cpus is None:
            total_cpus = os.cpu_count() or 4

        #logging.debug(f"Starting dynamic opt with {total_cpus} CPUs")

        tasks = []
        for i in range(self.positions.shape[0]):
            active_mask = (self.numbers[i] != 0)
            n_atoms = int(active_mask.sum().item())

            nums_np = self.numbers[i, :n_atoms].detach().cpu().numpy()
            pos_np = self.positions[i, :n_atoms, :].detach().cpu().numpy()

            # determine Threads
            n_threads = self._get_optimal_threads(n_atoms, total_cpus)

            tasks.append((nums_np, pos_np, i, "float64", self.gfn, n_threads))
        #scheduler
        results = self._dynamic_submitter(tasks, _run_single_opt_external, total_cpus)
        results.sort(key=lambda x: x[0])

        converged_list = []
        numbers_list = []
        positions_list = []
        hl_gap_list = []
        failure_reason_list = []

        for _, is_conv, res_num, res_pos, hl_gap, failure_reason in results:
            converged_list.append(is_conv)

            n_tens = torch.tensor(res_num, dtype=torch.int64, device=self.device)
            p_tens = torch.tensor(res_pos, dtype=__dtype__, device=self.device)

            numbers_list.append(n_tens)
            positions_list.append(p_tens)

            hl_gap_list.append(hl_gap)
            failure_reason_list.append(failure_reason)

        self.numbers = mctc.batch.pack(numbers_list)
        self.positions = mctc.batch.pack(positions_list)
        hl_gaps = np.array(hl_gap_list)

        return np.array(converged_list), hl_gaps, failure_reason_list

    def terahertz_upconversion_external(self, total_cpus=None):
        """
        Parallel terahertz upconversion with dynamic CPU allocation.
        """
        if total_cpus is None:
            total_cpus = os.cpu_count() or 4

        #logging.debug(f"Starting dynamic upconversion with {total_cpus} CPUs")

        tasks = []
        unpadded_lengths = {}

        for i in range(self.positions.shape[0]):
            active_mask = (self.numbers[i] != 0)
            n_atoms = int(active_mask.sum().item())
            unpadded_lengths[i] = n_atoms

            nums_np = self.numbers[i, :n_atoms].detach().cpu().numpy()
            pos_np = self.positions[i, :n_atoms, :].detach().cpu().numpy()

            # Determine Threads
            n_threads = self._get_optimal_threads(n_atoms, total_cpus)
            # Pack args: (nums, pos, i, unpadded_length, n_threads)
            tasks.append((nums_np, pos_np, i, self.gfn, n_threads))

        # Run Scheduler
        results = self._dynamic_submitter(tasks, _run_single_terahertz_external, total_cpus)

        # Sort and Pack
        results.sort(key=lambda x: x[0])

        P_values = []
        valid_indices = []

        for idx, valid, value in results:
            if valid:
                valid_indices.append(idx)

            P_values.append(value)

        # Update valid indices mask
        current_valid_mask = np.zeros(self.positions.shape[0], dtype=bool)
        current_valid_mask[valid_indices] = True
        self.valid_batch_indices = self.valid_batch_indices[current_valid_mask]

        P_values = np.array(P_values)

        return P_values


    def write_geometries_to_xyz(self, path, prefix = None):
        """
        Write the geometry(ies) to an xyz file.

        Args:
            path (str): Path to save the xyz file.
            prefix (str): Prefix for the filename.

        Raises:
            ValueError: If the positions tensor is not 2D or 3D.
        """

        #if positions is three dimensional -> batching
        if len(self.positions.shape) == 3:
            for i in range(self.positions.shape[0]):
                numbers = self.numbers[i].detach().numpy()
                atoms = [ATOM_DICT_ANUM[n][0].capitalize() for n in numbers]
                positions_T = self.positions[i].detach().numpy().T
                #convert to Angstrom
                positions_T = positions_T / AA2AU

                out = np.zeros((positions_T.shape[0] + 1, positions_T.shape[1]), dtype=object)
                out[0, :] = atoms
                out[1:, :] = positions_T
                if prefix is None:
                    filename = f"{path}/coord_opt_{i}.xyz"
                else:
                    filename = f"{path}/{prefix}_coord_opt_{i}.xyz"
                write_xyz_file(filename, out)

        elif len(self.positions.shape) == 2:
            numbers = calculator.numbers.detach().numpy()
            atoms = [ATOM_DICT_ANUM[n][0].capitalize() for n in numbers]
            positions_T = calculator.positions.detach().numpy().T
            # convert to Angstrom
            positions_T = positions_T / AA2AU

            out = np.zeros((positions_T.shape[0] + 1, positions_T.shape[1]), dtype=object)
            out[0, :] = atoms
            out[1:, :] = positions_T
            if prefix is None:
                filename = f"{path}/coord_opt.xyz"
            else:
                filename = f"{path}/{prefix}_coord_opt.xyz"
            write_xyz_file(filename, out)
        else:
            raise ValueError("positions must be either 2D or 3D tensor")






if __name__ == "__main__":
    smiles = ["C1=CC=CC=C1", "C1=CC=CC=C1"]
    #import rdkit.Chem as Chem
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    electronic_structure_calculator = Electronic_Structure_Calculator.from_mol_batch(mols)
    pass