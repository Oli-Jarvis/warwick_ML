import time

import numpy as np
import torch
import hashlib
import os
import logging
import matplotlib.pyplot as plt
from tad_mctc.io.write import write
from tad_mctc import read
from rdkit import Chem
from datetime import datetime
import tad_mctc as mctc
import io
import h5py
import time


from .electronic_structure import Electronic_Structure_Calculator
from .electronic_transport import  Electronic_Transport_Calculator_torch
from .phononic_transport import Phononic_Transport_Estimator_torch
from .constants import __Ha2eV__, __e0__, __hP__, __G0__
from .utils import print_vram_usage, align_molecule, add_gold, find_anchor_atom_indices, append_batch_to_xyz, append_to_hdf5, fig_to_numpy, ANG2BOHR, write_plot_data

__dtype__ = torch.float64
LOGGING_LEVEL = logging.ERROR
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

def _write_candidate_metadata(h5_filepath, hash_values, encoding_batch, failure_reasons, calculated_rewards, meta_data, fitness=None):
    hl_gaps = calculated_rewards.get("hl_gaps", np.full(len(encoding_batch), -1.0))
    smiles_anchored = calculated_rewards.get("smiles", [])
    oracle_call_start = meta_data.get("oracle_call_start", -1)
    for i in range(len(encoding_batch)):
        try:
            hl_gap_val = float(hl_gaps[i])
        except (TypeError, ValueError):
            hl_gap_val = -1.0
        attrs = {
            "encoding": str(encoding_batch[i]),
            "failure_reason": str(failure_reasons[i]),
            "hl_gap": hl_gap_val,
            "smiles": str(smiles_anchored[i]) if i < len(smiles_anchored) else "",
        }
        if oracle_call_start >= 0:
            attrs["oracle_call"] = int(oracle_call_start + i)
        _skip_keys = {"hash_values", "smiles", "failure_reasons", "oracle_calls", "hl_gaps"}
        datasets_to_store = {}
        for key, vals in calculated_rewards.items():
            if key in _skip_keys:
                continue
            try:
                val = vals[i]
            except (IndexError, KeyError):
                continue
            if isinstance(val, np.ndarray) and val.ndim > 0:
                datasets_to_store[key] = val
            else:
                try:
                    attrs[key] = float(val)
                except (TypeError, ValueError):
                    pass
        if fitness is not None:
            attrs["fitness"] = float(fitness[i])
        append_to_hdf5(h5_filepath, f"{hash_values[i]}/metadata", datasets_to_store, attributes=attrs)

def transport_workflow_handler(mols, encoding_batch, config_dict, meta_data = {},
                     calculated_rewards = None, batch_indices = None, anchor_atom = 16, anchor_mode="AuS_prepare", worker_id = 0):
    """
    Relax molecules, calculate phonon transport and electron transport

    Args:
        mols: list of rdkit mol objects
        encoding_batch: list of encoding strings (GGS or SMILES)
        config_dict: dictionary with configuration parameters
        meta_data: dictionary with meta data -> debug mode can be activated here
        calculated_rewards: dictionary with pre-calculated rewards -> each key is a numpy array
        batch_indices: indices of the batch to be processed
        anchor_atom: atomic number of the anchor atom to align the molecules to -> default is sulfur (16)
        anchor_mode: mode for aligning the molecules to the anchor atoms -> default is "AuS_prepare"
        worker_id: id of the worker process -> used for logging and storing data (not relevant as the parallelization is currently different.)

    Returns:
        calculated_rewards: dictionary with calculated rewards -> each key is a numpy array
        batch_indices: indices of the batch that was successfully processed (some molecules might be removed due to failed geometry optimization or invalid junction geometries)
        failure_reasons: list of strings with failure reasons for each molecule in the original batch (empty string if no failure)
    """

    worker_id = os.getpid()

    logger = setup_worker_logging()

    calculated_props = config_dict["calculated_props"]
    log_dir = config_dict["log_dir"]

    #this might be needed once the relaxation is done internally by dxtb
    #gradient_tolerance = config_dict["gradient_tolerance"]
    #fmax_tolerance = config_dict["fmax_tolerance"]
    #steps = config_dict["optimization_steps"]

    if "n_cpus" in config_dict.keys():
        n_cpus = config_dict["n_cpus"]
    else:
        n_cpus = os.cpu_count()
        print(f"Warning: n_cpus not specified in config_dict, using max available {n_cpus=}")

    logger.debug(f"batch_indices beginning={batch_indices}")

    #this is for the a full dxtb calculation on a gpu.
    if "memory_fraction_limit" in config_dict.keys() and torch.cuda.is_available():
        memory_fraction_limit = config_dict["memory_fraction_limit"]
        torch.cuda.set_per_process_memory_fraction(memory_fraction_limit, device=0)


    if "el_transp" in calculated_props:
        E_min = config_dict["E_min"]
        E_max = config_dict["E_max"]
        N_E_points = config_dict["N_E_points"]
        electrode_path = config_dict["electrode_path"]
    if "key_to_be_filled" in config_dict.keys():
        key_to_be_filled = config_dict["key_to_be_filled"]
    else:
        key_to_be_filled = []

    if batch_indices is None:
        batch_indices = np.arange(len(encoding_batch))
    if calculated_rewards is None:
        calculated_rewards = {}
    mols = np.array(mols)
    mols = mols[batch_indices]

    timestamp_ns = time.time_ns()
    hash_values = np.array([
        hashlib.md5(f"{encoding}|{timestamp_ns}|{i}".encode("utf-8")).hexdigest()
        for i, encoding in enumerate(encoding_batch)
    ])
    calculated_rewards["hash_values"] = hash_values
    failure_reasons = [""] * len(encoding_batch)

    h5_filename = f"worker_{worker_id}_data.h5"
    h5_filepath = os.path.join(config_dict["log_dir"], h5_filename)

    dump_filename_initial = f"initial_dump_worker_{worker_id}.xyz"
    dump_filename_full = f"full_dump_worker_{worker_id}.xyz"
    dump_filename_relaxed = f"relaxed_dump_worker_{worker_id}.xyz"
    dump_filename_full = os.path.join(log_dir, dump_filename_full)
    dump_filepath_relaxed = os.path.join(log_dir, dump_filename_relaxed)

    # align the molecules to sulfur-sulfur, shift gold atoms
    electronic_structure_calculator = Electronic_Structure_Calculator.from_mol_batch(mols)
    numbers = electronic_structure_calculator.numbers
    positions = electronic_structure_calculator.positions

    left_indices, right_indices = find_anchor_atom_indices(numbers, anchor_atom = anchor_atom)
    positions = align_molecule(positions, numbers, np.array([0, 0, 1]), left_indices, right_indices,
                               anchor_mode=anchor_mode, anchor_atom = anchor_atom)

    logger.debug("starting geometry optimization")

    optimize_geometry = True
    if optimize_geometry:
        # optimize the geometry of the molecules --> right now this is done externally because dxtb is not stable enough yet
        electronic_structure_calculator.positions = positions.contiguous()
        converged, hl_gaps, opt_failure_reasons = electronic_structure_calculator.optimize_geometry_external(total_cpus = n_cpus)

        failed_local = np.where(~converged)[0]
        for j in failed_local:
            failure_reasons[batch_indices[j]] = opt_failure_reasons[j]

        batch_indices = batch_indices[converged]

        full_hl_gaps = np.zeros(len(encoding_batch))
        full_hl_gaps[batch_indices] = hl_gaps[converged]
        calculated_rewards["hl_gaps"] = full_hl_gaps

        electronic_structure_calculator.turn_off_gradients()
        torch.cuda.empty_cache()  #
        numbers = electronic_structure_calculator.numbers[converged]
        positions = electronic_structure_calculator.positions[converged]

        del electronic_structure_calculator


    # edge case: if no valid molecules are left after filtering
    if len(batch_indices) == 0:
        logger.debug("no valid molecules left after geometry optimization")
        for key in key_to_be_filled:
            if key not in calculated_rewards.keys():
                calculated_rewards[key] = [0] * len(encoding_batch)
        return calculated_rewards, batch_indices, failure_reasons

    left_indices, right_indices = find_anchor_atom_indices(numbers, anchor_atom=79)
    positions = align_molecule(positions, numbers, np.array([0, 0, 1]), left_indices, right_indices)

    logger.debug("geometry optimization and alignment done")

    check_reasonable_junction_geom = True
    # check if molecules can form reasonable junction geometries -> if not, remove them from the batch
    # valid are molecules where no atoms are below the z position of the left anchor or above the z position of the right anchor
    # (with a tolerance of 0.25 angstrom)
    if check_reasonable_junction_geom:

        tolerance = 0.25 / ANG2BOHR
        batch_size = positions.shape[0]
        batch_range = torch.arange(batch_size, device=positions.device)
        z_coords = positions[:, :, 2]
        z_left = z_coords[batch_range, left_indices]
        z_right = z_coords[batch_range, right_indices]

        z_min = (z_left - tolerance).unsqueeze(1)
        z_max = (z_right + tolerance).unsqueeze(1)

        is_real_atom = (numbers != 0)
        too_low = (z_coords < z_min) & is_real_atom
        too_high = (z_coords > z_max) & is_real_atom

        is_invalid_molecule = (too_low | too_high).any(dim=1)
        logger.debug(f"invalid molecule found which cannon form reasonable junction geometries {is_invalid_molecule}")
        valid_geometry_mask = ~is_invalid_molecule

        invalid_junction_local = np.where(is_invalid_molecule.cpu().numpy())[0]
        for j in invalid_junction_local:
            failure_reasons[batch_indices[j]] = "invalid junction geometry"

        batch_indices = batch_indices[valid_geometry_mask.cpu().numpy()]

        # Update tensors
        numbers = numbers[valid_geometry_mask]
        positions = positions[valid_geometry_mask]
        left_indices = left_indices[valid_geometry_mask]
        right_indices = right_indices[valid_geometry_mask]

        # handle edge case where no batches are left
        if len(batch_indices) == 0:
            logger.debug("no valid molecules left after geometry optimization")
            for key in key_to_be_filled:
                if key not in calculated_rewards.keys():
                    calculated_rewards[key] = [0] * len(encoding_batch)
            _write_candidate_metadata(h5_filepath, hash_values, encoding_batch, failure_reasons, calculated_rewards, meta_data)
            return calculated_rewards, batch_indices, failure_reasons


    if "ph_transp" in calculated_props:
        active_masks = [numbers[i] != 0 for i in range(numbers.shape[0])]
        active_numbers = [numbers[i][active_masks[i]] for i in range(numbers.shape[0])]
        unpadded_lengths = np.array([len(active_numbers[i]) for i in range(numbers.shape[0])])
        unpadded_lengths = np.max(unpadded_lengths)
        electronic_structure_calculator = Electronic_Structure_Calculator(numbers[:, :unpadded_lengths],
                                                                          positions[:, :unpadded_lengths])

        logger.debug("Calculating hessian")
        logger.debug(f"batch_indices={batch_indices}")
        #calculate hessian externally as dxtb is not stable enough yet
        hessian, hess_failure_reasons = electronic_structure_calculator.get_hessian_external_xtb(total_cpus = n_cpus)
        valid_local_indices = electronic_structure_calculator.valid_batch_indices
        for j, reason in enumerate(hess_failure_reasons):
            if reason:
                failure_reasons[batch_indices[j]] = reason
        #check if at least one hessian failed -> then new electronic structure calculator has to be created
        one_hessian_failed = (len(valid_local_indices) != len(batch_indices))
        batch_indices = batch_indices[valid_local_indices]
        logger.debug(f"batch_indices handler {batch_indices}")
        if one_hessian_failed:
            logger.debug(f"one hessian failed {one_hessian_failed}, {batch_indices}. recreating electronic structure calculator")
            numbers = numbers[valid_local_indices]
            positions = positions[valid_local_indices]
            hessian = hessian[valid_local_indices]
            left_indices = left_indices[valid_local_indices]
            right_indices = right_indices[valid_local_indices]

            # handle edge case where largest batch was removed -> now padding is wrong, has to be smaller
            new_padding_needed = torch.all(numbers[:, -1] == 0).cpu().detach()
            if new_padding_needed:
                numbers_list = []
                positions_list = []
                hessian_list = []

                for i in range(numbers.shape[0]):
                    #padding is at the end
                    k = (numbers[i] != 0).sum()

                    numbers_list.append(numbers[i, :k])
                    positions_list.append(positions[i, :k, :])
                    hessian_list.append(hessian[i, :3*k, :3*k])

                if len(batch_indices) > 0:
                    numbers = mctc.batch.pack(numbers_list).to(numbers.device)
                    positions = mctc.batch.pack(positions_list).to(numbers.device)
                    hessian = mctc.batch.pack(hessian_list).to(numbers.device)

            # handle edge case where no batches are left
            if len(batch_indices) == 0:
                logger.debug("no valid molecules left after geometry optimization")
                for key in key_to_be_filled:
                    if key not in calculated_rewards.keys():
                        calculated_rewards[key] = [0] * len(encoding_batch)
                _write_candidate_metadata(h5_filepath, hash_values, encoding_batch, failure_reasons, calculated_rewards, meta_data)
                return calculated_rewards, batch_indices, failure_reasons



            electronic_structure_calculator = Electronic_Structure_Calculator(numbers, positions)

            phonon_transport_calculator = Phononic_Transport_Estimator_torch(electronic_structure_calculator, 20, 5000, hessian = hessian)
        else:
            phonon_transport_calculator = Phononic_Transport_Estimator_torch(electronic_structure_calculator, 20, 5000, hessian = hessian)

        logger.debug("calculating phonon transport")

        tau_ph_calculated = phonon_transport_calculator.calculate_tau_ph()
        tau_ph_calculated = tau_ph_calculated.cpu().detach().numpy()
        E = phonon_transport_calculator.E.cpu().detach().numpy()
        tau_ph = np.zeros((len(encoding_batch), E.shape[0]))
        tau_ph[batch_indices] = tau_ph_calculated

        kappa_ph = np.ones(len(encoding_batch)) * 0
        kappa_ph_calculated = phonon_transport_calculator.calculate_kappa_ph().cpu().detach().numpy()
        kappa_ph[batch_indices] = kappa_ph_calculated
        calculated_rewards["k_ph"] = np.array(kappa_ph)

        del phonon_transport_calculator
        del electronic_structure_calculator
        torch.cuda.empty_cache()

        append_batch_to_xyz(dump_filepath_relaxed, numbers, positions, hash_values, batch_indices, worker_id,"relaxed")

        # store raw data and plot
        counter = 0
        for i in range(len(encoding_batch)):
            if i in batch_indices:
                # Prep data
                mol_nums = numbers[counter]
                mol_pos = positions[counter]
                mask = mol_nums != 0
                real_nums = mol_nums[mask].cpu().numpy()
                real_pos = mol_pos[mask].cpu().numpy() / ANG2BOHR
                data_to_store = {
                    "energy_eV": E,
                    "transmission": tau_ph[i],
                    "atomic_numbers": real_nums,
                    "positions": real_pos
                }

                if "debug" in list(meta_data.keys()) and meta_data["debug"] == True:
                    fig = plt.figure()
                    plt.plot(E, tau_ph[i])
                    plt.yscale("log")
                    plt.title(f"Phonon: {hash_values[i]}")

                    # Convert plot to binary array
                    plot_blob = fig_to_numpy(fig)
                    data_to_store["plot_png"] = plot_blob  # Store the PNG bytes
                    plt.close(fig)

                group_path = f"{hash_values[i]}/phonon"
                append_to_hdf5(h5_filepath, group_path, data_to_store,
                               attributes={"kappa": kappa_ph[i]})
                counter += 1

        #uncomment this if you want an overview plot
        #for i in range(len(group_selfies_batch)):
        #    plt.plot(E, tau_ph[i], label = calculated_rewards["k_ph"][i])
        #plt.yscale("log")
        #plt.ylim(1E-6, 3)
        #plt.legend()
        #plt.savefig(f"{log_dir}/tau_ph.svg")
        #plt.clf()

    if "el_transp" in calculated_props:
        logger.debug("starting electronic transport calculation")
        # add gold electrodes
        if torch.cuda.is_available():
            dd = {"dtype": __dtype__, "device": torch.device("cuda:0")}
        else:
            dd = {"dtype": __dtype__, "device": torch.device("cpu")}


        num_left, pos_left = read(f"{electrode_path}/Au_111_3x4x6_left_top.xyz", **dd)
        num_right, pos_right = read(f"{electrode_path}/Au_111_3x4x6_right_top.xyz", **dd)
        numbers, positions, valid_geometry_mask = add_gold(numbers, positions, left_indices, right_indices,
                                                           num_left, pos_left,
                                                           num_right, pos_right, check_valid=True)

        valid_geometry_mask = valid_geometry_mask.cpu().detach().numpy()
        logger.debug(f"batch_indices valid geometry={batch_indices}, {valid_geometry_mask}")
        for j in np.where(~valid_geometry_mask)[0]:
            failure_reasons[batch_indices[j]] = "invalid junction geometry with gold electrodes"
        batch_indices = batch_indices[valid_geometry_mask]
        logger.debug(f"batch_indices={batch_indices}")

        if len(batch_indices) == 0:
            for key in key_to_be_filled:
                if key not in calculated_rewards.keys():
                    calculated_rewards[key] = [0] * len(encoding_batch)
            _write_candidate_metadata(h5_filepath, hash_values, encoding_batch, failure_reasons, calculated_rewards, meta_data)
            return calculated_rewards, batch_indices, failure_reasons

        numbers = numbers[valid_geometry_mask]
        positions = positions[valid_geometry_mask]

        #store full geometry data
        append_batch_to_xyz(dump_filename_full, numbers, positions, hash_values, batch_indices, worker_id,"full")

        g_surf_left_path = f"{electrode_path}/left"
        g_surf_right_path = f"{electrode_path}/right"

        electronic_structure_calculator = Electronic_Structure_Calculator(numbers, positions)
        retry_counter = 0
        max_retries = 10
        coefficients_valid = False
        #dxtb is not stable yet. The following part is a workaround to check if the coefficients are valid and if not,
        # remove the failed batches and recalculate until it works or max retries is reached. The error message is checked
        # for "Fermi energy" to make sure that only this error is caught and not other errors that might occur.
        while coefficients_valid == False:
            try:
                print(f"Fermi {retry_counter=}")
                retry_counter += 1
                if retry_counter > max_retries:
                    logger.debug("fermi retry loop does not work")
                    for global_i in batch_indices:
                        failure_reasons[global_i] = "Fermi energy convergence failed: max retries exceeded"
                    for key in key_to_be_filled:
                        if key not in calculated_rewards.keys():
                            calculated_rewards[key] = [0] * len(encoding_batch)
                    _write_candidate_metadata(h5_filepath, hash_values, encoding_batch, failure_reasons, calculated_rewards, meta_data)
                    return calculated_rewards, batch_indices, failure_reasons

                electronic_structure_calculator.coefficients
                coefficients_valid = True
            except RuntimeError as e:
                print("message full ecc:", e.args[0])
                if "Fermi energy" not in e.args[0]:
                    raise RuntimeError(e)

                value = e.args[1]
                # check if not all values are 0 along the first axis
                valid_batches_mask = torch.any(value, dim=(1, 2)).cpu().detach().numpy()
                # occupation of the highest orbital should be 0
                is_last_entry_one_alpha = value[:, 0, -1] != 0
                is_last_entry_one_beta = value[:, 1, -1] != 0
                invalid_batches_mask = is_last_entry_one_alpha | is_last_entry_one_beta
                invalid_batches_mask = invalid_batches_mask.cpu().detach().numpy()
                # combine the masks
                valid_batches_mask = valid_batches_mask & ~invalid_batches_mask
                # check if all true
                if np.all(valid_batches_mask):
                    raise ValueError(
                        "Fermi energy calculation failed but all batches are valid. This should not happen.")

                for j in np.where(~valid_batches_mask)[0]:
                    failure_reasons[batch_indices[j]] = "Fermi energy convergence failed"

                numbers = numbers[valid_batches_mask]
                positions = positions[valid_batches_mask]
                batch_indices = batch_indices[valid_batches_mask]

                #handle edge case where largest batch was removed -> now padding is wrong, has to be smaller
                new_padding_needed = torch.all(numbers[:, -1] == 0).cpu().detach()
                if new_padding_needed:
                    numbers_list = []
                    positions_list = []
                    for i in range(numbers.shape[0]):
                        is_real_atom = (numbers[i] != 0)
                        numbers_list.append(numbers[i][is_real_atom])
                        positions_list.append(positions[i][is_real_atom])
                    if len(batch_indices) > 0:
                        numbers = mctc.batch.pack(numbers_list).to(numbers.device)
                        positions = mctc.batch.pack(positions_list).to(numbers.device)

                #handle edge case where no batches are left
                if len(batch_indices) == 0:
                    logger.debug("no valid molecules left after geometry optimization")
                    for key in key_to_be_filled:
                        if key not in calculated_rewards.keys():
                            calculated_rewards[key] = [0] * len(encoding_batch)
                    _write_candidate_metadata(h5_filepath, hash_values, encoding_batch, failure_reasons, calculated_rewards, meta_data)
                    return calculated_rewards, batch_indices, failure_reasons


                electronic_structure_calculator = Electronic_Structure_Calculator(numbers, positions)

        calculator = Electronic_Transport_Calculator_torch(electronic_structure_calculator, E_min, E_max,
                                                           N_E_points, g_surf_left_path,
                                                           g_surf_right_path, WBL=False)

        tau_el = np.zeros((len(encoding_batch), N_E_points))
        G_el = np.zeros(len(encoding_batch))
        S_el = np.zeros(len(encoding_batch))
        k_el = np.zeros(len(encoding_batch))

        tau_el[batch_indices] = calculator.tau_el.cpu().detach().numpy()
        G_el[batch_indices] = calculator.G_el.cpu().detach().numpy()
        S_el[batch_indices] = calculator.S_el.cpu().detach().numpy()
        k_el[batch_indices] = calculator.kappa_el.cpu().detach().numpy()
        E = calculator.E.cpu().detach().numpy() * __Ha2eV__ - calculator.E_fermi

        ZT_300K = np.zeros(len(encoding_batch))
        if "k_ph" in calculated_rewards:
            kph = np.array(calculated_rewards["k_ph"])
            denom = kph + k_el
            nonzero = denom > 0
            ZT_300K[nonzero] = G_el[nonzero] * S_el[nonzero]**2 / denom[nonzero] * (300.0 * 7.748091729e-5)
        calculated_rewards["ZT"] = ZT_300K

        counter = 0
        for i in range(len(encoding_batch)):
            if i in batch_indices:
                mol_nums = numbers[counter]
                mol_pos = positions[counter]
                mask = mol_nums != 0
                real_nums = mol_nums[mask].cpu().numpy()
                real_pos = mol_pos[mask].cpu().numpy() / ANG2BOHR

                data_to_store = {
                    "energy_fermi_shifted": E,
                    "transmission": tau_el[i],
                    "atomic_numbers": real_nums,
                    "positions": real_pos
                }

                if "debug" in list(meta_data.keys()) and meta_data["debug"] == True:
                    fig = plt.figure()
                    plt.plot(E, tau_el[i])
                    plt.yscale("log")
                    plt.title(f"Electronic: {hash_values[i]}")

                    plot_blob = fig_to_numpy(fig)
                    data_to_store["plot_png"] = plot_blob
                    plt.close(fig)

                # write
                group_path = f"{hash_values[i]}/electronic"
                append_to_hdf5(h5_filepath, group_path, data_to_store,
                               attributes={"G": G_el[i], "S": S_el[i], "k_el": k_el[i], "ZT": ZT_300K[i]})
                counter += 1

        logger.debug("Electronic transport calculation done")

        calculated_rewards["G"] = G_el
        calculated_rewards["S"] = S_el
        calculated_rewards["k_el"] = k_el

        del calculator
        del electronic_structure_calculator
        torch.cuda.empty_cache()



        #if "debug" in list(meta_data.keys()) and meta_data["debug"] == True:
        #    counter = 0
        #    for i, b in enumerate(batch_indices):
        #        data_dir = f"{log_dir}/full_data/{hash_values[b]}"
        #        plt.plot(E, tau_el[b])
        #        plt.yscale("log")
        #        plt.xlabel(r"$E-E_\mathrm{F}$ (eV)")
        #        plt.ylabel("el. Transmission")
        #        plt.axvline(0, color="black", linestyle="--")
        #        plt.savefig(f"{data_dir}/el_transmission.pdf")
        #        plt.savefig(f"{data_dir}/el_transmission.svg")
        #        write_plot_data(f"{data_dir}/el_transmission.txt", [E, tau_el[b]])
        #        plt.clf()

            #uncomment this if you want an overview plot
            #for i, b in enumerate(batch_indices):
            #    plt.plot(E, tau_el[b], label = i)
            #plt.legend()
            #plt.yscale("log")
            #plt.xlabel(r"$E-E_\mathrm{F}$ (eV)")
            #plt.ylabel("el. Transmission")
            #plt.axvline(0, color="black", linestyle="--")
            #plt.savefig(f"{log_dir}/el_transmission_overview.pdf")
            #plt.savefig(f"{log_dir}/el_transmission_overview.svg")
            #plt.clf()


    return calculated_rewards, batch_indices, failure_reasons

if __name__ == '__main__':
    pass

