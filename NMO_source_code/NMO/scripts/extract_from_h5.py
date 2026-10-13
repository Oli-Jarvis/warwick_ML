import argparse
from pathlib import Path
import h5py
import matplotlib.pyplot as plt


def extract_geometry(target_hash: str, dump_path: Path, output_path: Path) -> bool:
    if not dump_path.exists():
        print(f"Warning: Dump file not found at {dump_path}")
        return False

    print(f"Searching for hash '{target_hash}' in {dump_path.name}...")

    with open(dump_path, 'r') as f_in:
        while True:
            line_num_atoms = f_in.readline()
            if not line_num_atoms:
                break

            try:
                num_atoms = int(line_num_atoms.strip())
            except ValueError:
                continue

            line_comment = f_in.readline()
            atom_lines = [f_in.readline() for _ in range(num_atoms)]

            if f"hash={target_hash}" in line_comment:
                with open(output_path, 'w') as f_out:
                    f_out.write(line_num_atoms)
                    f_out.write(line_comment)
                    f_out.writelines(atom_lines)
                print(f"-> Geometry saved to: {output_path}")
                return True

    print(f"-> Hash '{target_hash}' not found in {dump_path.name}.")
    return False


def plot_transmissions(target_hash: str, h5_path: Path, output_image: Path, mode: str):
    if not h5_path.exists():
        print(f"Error: HDF5 file not found at {h5_path}")
        return

    try:
        with h5py.File(h5_path, 'r') as f:
            if target_hash not in f:
                print(f"Error: Hash '{target_hash}' not found in HDF5.")
                return

            plt.style.use('bmh')

            # Setup plots based on mode
            if mode == "ZT":
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                ax_el, ax_ph = ax[0], ax[1]
            else:  # k_ph
                fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                ax_ph = ax
                ax_el = None

            # --- Electronic Plot (ZT mode only) ---
            if mode == "ZT":
                if f"{target_hash}/electronic" in f:
                    el_group = f[f"{target_hash}/electronic"]
                    E_el = el_group['energy_fermi_shifted'][:]
                    tau_el = el_group['transmission'][:]

                    ax_el.plot(E_el, tau_el, linewidth=1.2)
                    ax_el.set_yscale('log')
                    ax_el.set_xlabel(r'$E - E_F$ (eV)')
                    ax_el.set_ylabel('Transmission ($G/G_0$)')
                    ax_el.set_title('Electronic Transmission')
                else:
                    ax_el.text(0.5, 0.5, "No Electronic Data", ha='center', va='center', transform=ax_el.transAxes)

            # --- Phononic Plot (ZT and k_ph modes) ---
            if f"{target_hash}/phonon" in f:
                ph_group = f[f"{target_hash}/phonon"]
                E_ph = ph_group['energy_eV'][:]
                tau_ph = ph_group['transmission'][:]

                ax_ph.plot(E_ph, tau_ph, color='tab:orange', linewidth=1.2)
                ax_ph.set_yscale('log')
                ax_ph.set_xlabel('Energy (eV)')
                ax_ph.set_ylabel('Transmission')
                ax_ph.set_title('Phononic Transmission')
            else:
                ax_ph.text(0.5, 0.5, "No Phonon Data", ha='center', va='center', transform=ax_ph.transAxes)

            plt.tight_layout()
            plt.savefig(output_image)
            print(f"-> Plot saved to: {output_image}")
            plt.close()

    except Exception as e:
        print(f"An error occurred reading the HDF5 file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extract geometry and plot transmission.")
    parser.add_argument("hash", type=str, help="The hash value of the molecule.")
    parser.add_argument("--mode", type=str, choices=["ZT", "k_ph", "upconversion"], default="ZT",
                        help="Mode: 'ZT' (Electronic+Phonon), 'k_ph' (Phonon only), or 'upconversion' (Extract initial/aligned/relaxed geometries, no plot).")
    parser.add_argument("--dir", type=str, default=".", help="Directory containing log files.")
    parser.add_argument("--filename", type=str, default="worker_0_data.h5", help="HDF5 filename (e.g. worker_12345_data.h5).")

    args = parser.parse_args()

    base_dir = Path(args.dir)
    target_hash = args.hash
    filename = args.filename

    # --- Mode: Upconversion (Extract 3 geometries, No Plot) ---
    if args.mode == "upconversion":
        stages = ["initial", "aligned", "relaxed"]
        for stage in stages:
            dump_file = base_dir / f"{stage}_dump_{filename}"
            output_xyz = base_dir / f"{target_hash}_{stage}.xyz"
            extract_geometry(target_hash, dump_file, output_xyz)

    # --- Mode: ZT or k_ph (Extract relaxed geometry + Plot) ---
    else:
        # Standard extraction (relaxed only)
        xyz_dump = base_dir / f"relaxed_dump_{filename}"
        output_xyz = base_dir / f"{target_hash}.xyz"
        extract_geometry(target_hash, xyz_dump, output_xyz)

        # Plotting
        h5_file = base_dir / filename
        output_plot = base_dir / f"plot_{target_hash}.svg"
        plot_transmissions(target_hash, h5_file, output_plot, args.mode)


if __name__ == "__main__":
    main()