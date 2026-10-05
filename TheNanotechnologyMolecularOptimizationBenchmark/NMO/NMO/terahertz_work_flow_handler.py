
import numpy as np
import torch
import hashlib
import os
import logging
import matplotlib.pyplot as plt
from tad_mctc import read
from rdkit import Chem
from datetime import datetime
import time
import tad_mctc as mctc
from tad_mctc.io.write import write


from .electronic_structure import Electronic_Structure_Calculator
from .electronic_transport import  Electronic_Transport_Calculator_torch
from .phononic_transport import Phononic_Transport_Estimator_torch
from .constants import __Ha2eV__, __e0__, __hP__, __G0__
from .utils import print_vram_usage, align_molecule, add_gold, find_anchor_atom_indices, append_batch_to_xyz, append_to_hdf5, make_circle, ANG2BOHR
from .transport_workflow_handler import _write_candidate_metadata


__dtype__ = torch.float64
LOGGING_LEVEL = logging.DEBUG
def setup_worker_logging():
    pid = os.getpid()    

    logger_name = f"worker_{pid}"
    logger = logging.getLogger(logger_name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(LOGGING_LEVEL)    

    logger.propagate = False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"worker_{pid}_{timestamp}.log"   


    file_handler = logging.FileHandler(log_filename, delay=True)    
    
    formatter = logging.Formatter('%(asctime)s - PID:%(process)d - %(levelname)s - %(message)s')
    
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def terahertz_workflow_handler(mols, group_selfies_batch, config_dict, meta_data = {},
                     calculated_rewards = None, batch_indices = None, anchor_atoms = None, anchor_mode="AuS_prepare", worker_id = 0):
    """
    Relax molecules, calculate terahertz upconversion efficiency

    Args:
        mols: list of rdkit mol objects
        group_selfies_batch: list of group selfies strings
        config_dict: dictionary with configuration parameters
        meta_data: dictionary with meta data
        calculated_rewards: dictionary with pre-calculated rewards -> each key is a numpy array
        batch_indices: indices of the batch to be processed
        anchor_atoms: List of anchor atoms tuples for the alignment -> if None no alignment is performed
    """

    worker_id = os.getpid()

    logger = setup_worker_logging()    

    calculated_props = config_dict["calculated_props"]
    log_dir = config_dict["log_dir"]
    if "n_cpus" in config_dict.keys():
        n_cpus = config_dict["n_cpus"]
    else:
        n_cpus = os.cpu_count()
        print(f"Warning: n_cpus not specified in config_dict, using max available {n_cpus=}")

    logger.debug(f"batch_indices beginning={batch_indices} and {n_cpus=}")

    #create one dump file because hpc system has quota on number of files
    dump_filename_initial = f"initial_dump_worker_{worker_id}.xyz"
    dump_filename_aligned = f"aligned_dump_worker_{worker_id}.xyz"
    dump_filename_relaxed = f"relaxed_dump_worker_{worker_id}.xyz"
    dump_filepath_initial = os.path.join(log_dir, dump_filename_initial)
    dump_filepath_aligned = os.path.join(log_dir, dump_filename_aligned)
    dump_filepath_relaxed = os.path.join(log_dir, dump_filename_relaxed)


    #this is for the a full dxtb calaculation
    if "memory_fraction_limit" in config_dict.keys() and torch.cuda.is_available():
        memory_fraction_limit = config_dict["memory_fraction_limit"]
        torch.cuda.set_per_process_memory_fraction(memory_fraction_limit, device=0)

    if "key_to_be_filled" in config_dict.keys():
        key_to_be_filled = config_dict["key_to_be_filled"]
    else:
        key_to_be_filled = []

    if batch_indices is None:
        batch_indices = np.arange(len(group_selfies_batch))
    if calculated_rewards is None:
        calculated_rewards = {}

    failure_reasons = [""] * len(group_selfies_batch)

    h5_filename = f"worker_{worker_id}_data.h5"
    h5_filepath = os.path.join(log_dir, h5_filename)

    mols = np.array(mols)
    mols = mols[batch_indices]
    if anchor_atoms is not None:
        anchor_atoms = anchor_atoms[batch_indices]

    # align the molecules to sulfur-sulfur, shift gold atoms
    electronic_structure_calculator = Electronic_Structure_Calculator.from_mol_batch(mols, gfn = 2)
    numbers = electronic_structure_calculator.numbers
    positions = electronic_structure_calculator.positions

    logger.debug(f"starting geometry optimization {batch_indices=}")

    timestamp_ns = time.time_ns()
    hash_values = np.array([
        hashlib.md5(f"{encoding}|{timestamp_ns}|{i}".encode("utf-8")).hexdigest()
        for i, encoding in enumerate(group_selfies_batch)
    ])
    calculated_rewards["hash_values"] = hash_values


    if "debug" in list(meta_data.keys()) and meta_data["debug"] == True:
        append_batch_to_xyz(dump_filepath_initial, numbers, positions, hash_values, batch_indices, worker_id, "initial")

    optimize_geometry = True
    if optimize_geometry:
        # optimize the geometry of the molecules --> right now this is done externally because dxtb is not stable enough yet
        electronic_structure_calculator.positions = positions.contiguous()
        converged, hl_gaps, opt_failure_reasons = electronic_structure_calculator.optimize_geometry_external(total_cpus = n_cpus)

        failed_local = np.where(~converged)[0]
        for j in failed_local:
            failure_reasons[batch_indices[j]] = opt_failure_reasons[j]

        batch_indices = batch_indices[converged]

        full_hl_gaps = np.zeros(len(group_selfies_batch))
        full_hl_gaps[batch_indices] = hl_gaps[converged]
        calculated_rewards["hl_gaps"] = full_hl_gaps

        electronic_structure_calculator.turn_off_gradients()
        torch.cuda.empty_cache()
        numbers = electronic_structure_calculator.numbers[converged]
        positions = electronic_structure_calculator.positions[converged]
        if anchor_atoms is not None:
            anchor_atoms = anchor_atoms[converged]

        del electronic_structure_calculator


    # edge case: if no valid molecules are left after filtering
    if len(batch_indices) == 0:
        logger.debug(f"no valid molecules left after geometry optimization {converged=}")
        for key in key_to_be_filled:
            if key not in calculated_rewards.keys():
                calculated_rewards[key] = [0] * len(group_selfies_batch)
        _write_candidate_metadata(h5_filepath, hash_values, group_selfies_batch, failure_reasons, calculated_rewards, meta_data)
        return calculated_rewards, batch_indices, failure_reasons


    check_reasonable_junction_geom = True
    if anchor_atoms is not None:
        print(anchor_atoms)
        positions = align_molecule(positions, numbers, np.array([0, 0, 1]), left_atom_indices=anchor_atoms[:, 0], right_atom_indices=anchor_atoms[:, 1], anchor_mode = "")
    else:
        check_reasonable_junction_geom = False


    if "debug" in list(meta_data.keys()) and meta_data["debug"] == True:
        append_batch_to_xyz(dump_filepath_aligned, numbers, positions, hash_values, batch_indices, worker_id, "aligned")


    logger.debug("geometry optimization and alignment done")

    
    # check if molecules can form reasonable SAM structure -> if not, remove them from the batch
    # valid are molecules where no atoms are below the z position of the left anchor
    # (with a tolerance of 0.25 angstrom)
    if check_reasonable_junction_geom:

        tolerance = 0.25 / ANG2BOHR
        batch_size = positions.shape[0]
        batch_range = torch.arange(batch_size, device=positions.device)
        z_coords = positions[:, :, 2]
        z_left = z_coords[batch_range, anchor_atoms[:, 0]]
        #find maximum z coordinate of right anchor atoms
        z_max = z_coords.max(dim=1)[0]

        #maximum extension of molecule in z direction beyond right anchor atom in angstrom
        # length penalty --> scales with 1/g^2. We use molecular length as a proxy for g
        molecular_lengths = (z_max - z_left).unsqueeze(1)/ANG2BOHR

        z_min = (z_left - tolerance).unsqueeze(1)
        #z_max = (z_right + tolerance).unsqueeze(1)

        is_real_atom = (numbers != 0)
        too_low = (z_coords < z_min) & is_real_atom
        #too_high = (z_coords > z_max) & is_real_atom

        is_invalid_molecule = too_low.any(dim=1)
        logger.debug(f"invalid molecule found which cannon form reasonable junction geometries {is_invalid_molecule}")
        valid_geometry_mask = ~is_invalid_molecule

        invalid_local = np.where(is_invalid_molecule.cpu().numpy())[0]
        for j in invalid_local:
            failure_reasons[batch_indices[j]] = "invalid SAM geometry"

        batch_indices = batch_indices[valid_geometry_mask.cpu().numpy()]

        # Update tensors
        numbers = numbers[valid_geometry_mask]
        positions = positions[valid_geometry_mask]
        molecular_lengths = molecular_lengths[valid_geometry_mask]

        # handle edge case where no batches are left
        if len(batch_indices) == 0:
            logger.debug("no valid molecules left after geometry optimization")
            for key in key_to_be_filled:
                if key not in calculated_rewards.keys():
                    calculated_rewards[key] = [0] * len(group_selfies_batch)
            _write_candidate_metadata(h5_filepath, hash_values, group_selfies_batch, failure_reasons, calculated_rewards, meta_data)
            return calculated_rewards, batch_indices, failure_reasons


    #create electronic structure calculator with new positions/numbers calculated above
    active_masks = [numbers[i] != 0 for i in range(numbers.shape[0])]
    active_numbers = [numbers[i][active_masks[i]] for i in range(numbers.shape[0])]
    unpadded_lengths = np.array([len(active_numbers[i]) for i in range(numbers.shape[0])])
    unpadded_lengths = np.max(unpadded_lengths)
    electronic_structure_calculator = Electronic_Structure_Calculator(numbers[:, :unpadded_lengths],
                                                                      positions[:, :unpadded_lengths],
                                                                      gfn=2)

    P_values_local = electronic_structure_calculator.terahertz_upconversion_external(total_cpus = n_cpus)

    valid_local_indices = electronic_structure_calculator.valid_batch_indices
    all_local_indices = np.arange(len(batch_indices))
    failed_upconv_local = np.setdiff1d(all_local_indices, valid_local_indices)
    for j in failed_upconv_local:
        failure_reasons[batch_indices[j]] = "terahertz upconversion calculation failed"

    P_values_local = P_values_local[valid_local_indices]
    molecular_lengths = molecular_lengths[valid_local_indices]
    numbers = numbers[valid_local_indices]
    positions = positions[valid_local_indices]
    batch_indices = batch_indices[valid_local_indices]

    # handle edge case where no batches are left
    if len(batch_indices) == 0:
        logger.debug("no valid molecules left after geometry optimization")
        for key in key_to_be_filled:
            if key not in calculated_rewards.keys():
                calculated_rewards[key] = [0] * len(group_selfies_batch)
        _write_candidate_metadata(h5_filepath, hash_values, group_selfies_batch, failure_reasons, calculated_rewards, meta_data)
        return calculated_rewards, batch_indices, failure_reasons

    P_values = np.ones(len(group_selfies_batch)) * 0
    log_P_values = np.ones(len(group_selfies_batch)) * -100000000
    log_P_values_length_scaled = np.ones(len(group_selfies_batch)) * -100000000
    P_values[batch_indices] = P_values_local
    log_P_values[batch_indices] = np.log10(P_values_local)

    # -----------------------------------------------------
    # penalty for surface area
    areas = []
    pos_np = positions.detach().cpu().numpy()
    nums_np = numbers.detach().cpu().numpy()

    for i in range(len(pos_np)):
        # Select valid atoms
        mask = nums_np[i] > 0
        coords = pos_np[i][mask]

        points_2d = coords[:, :2].tolist()
        center, radius_bohr = make_circle(points_2d)

        radius_ang = radius_bohr / ANG2BOHR
        area = np.pi * (radius_ang ** 2)

        area = max(area, 1e-5)
        areas.append(area)

    areas = np.array(areas)
    # -----------------------------------------------------

    sigma = 0.3809242121851108
    offset = 5
    #see if w_len and w_are are in config dict
    if "w_len" in config_dict:
        w_len = config_dict["w_len"]
    else:
        w_len = 1.0
    if "w_area" in config_dict:
        w_area = config_dict["w_area"]
    else:
        w_area = 1.0

    raw_score = (np.log10(P_values_local)
                - w_len * (2/sigma) * np.log10(molecular_lengths.cpu().numpy().flatten())
                - w_area * (1 / sigma) * np.log10(areas))
    exponent = np.clip(raw_score + offset, -50, 80)
    stable_reward = np.log10(1 + 10 ** exponent)
    log_P_values_length_scaled[batch_indices] = stable_reward

    calculated_rewards["P_upconversion"] = P_values
    calculated_rewards["log_P_upconversion"] = log_P_values
    calculated_rewards["log_P_upconversion_scaled"] = log_P_values_length_scaled

    full_molecular_lengths = np.zeros(len(group_selfies_batch))
    full_areas = np.zeros(len(group_selfies_batch))

    full_molecular_lengths[batch_indices] = molecular_lengths.detach().cpu().numpy().flatten()
    full_areas[batch_indices] = areas

    calculated_rewards["molecular_length"] = full_molecular_lengths
    calculated_rewards["surface_area"] = full_areas

    # log full data for debug
    append_batch_to_xyz(dump_filepath_relaxed, numbers, positions, hash_values, batch_indices, worker_id, "relaxed")

    # store per-molecule terahertz data to HDF5
    counter = 0
    for i in range(len(group_selfies_batch)):
        if i in batch_indices:
            mol_nums = numbers[counter]
            mol_pos = positions[counter]
            mask = mol_nums != 0
            real_nums = mol_nums[mask].cpu().numpy()
            real_pos = mol_pos[mask].cpu().numpy() / ANG2BOHR
            data_to_store = {
                "atomic_numbers": real_nums,
                "positions": real_pos,
            }
            group_path = f"{hash_values[i]}/terahertz"
            append_to_hdf5(h5_filepath, group_path, data_to_store,
                           attributes={})
            counter += 1

    return calculated_rewards, batch_indices, failure_reasons

