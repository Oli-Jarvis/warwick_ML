# Ensure you have ASE installed: pip install ase

import ase.build
import ase.io
import numpy as np

# --- Parameters ---
element = 'Au'          # Chemical symbol for Gold
surface_index = (1, 1, 1) # Miller index of the surface
size = (3, 4, 6)      # Supercell size (4x4 surface units, 6 layers thick)
lattice_constant = 4.07999881703924 # Approximate experimental lattice constant for bulk Au (FCC) in Angstroms
vacuum_space = 00.0     # Vacuum thickness in Angstroms added above the slab

# --- Create the Au(111) surface slab ---
# ASE's fcc111 builder is convenient for this common surface.
# 'orthogonal=True' generates a hexagonal surface unit cell which is often
# easier to work with than the primitive non-orthogonal cell.
# The 'size' parameter (nx, ny, nz) replicates the surface unit cell
# nx times along the first surface vector, ny times along the second,
# and creates nz layers.
# The 'vacuum' parameter adds empty space along the z-axis (perpendicular
# to the surface) on top of the slab to separate periodic images.

print(f"Generating Au{surface_index} surfaces...")
print(f"Supercell size: {size[0]}x{size[1]} surface units, {size[2]} layers.")
print(f"Using bulk lattice constant a = {lattice_constant} Å")
print(f"Adding {vacuum_space} Å vacuum on top.")

slab = ase.build.fcc111(symbol=element,
                        size=size,
                        a=lattice_constant,
                        periodic=True,
                        orthogonal=True) # Use orthogonal (hexagonal) unit cell


# --- Define output filenames ---
# Create descriptive filenames based on parameters
base_filename = f"{element}_{''.join(map(str, surface_index))}_{size[0]}x{size[1]}x{size[2]}_left"
poscar_filename = base_filename + ".poscar"
xyz_filename = base_filename + ".xyz"

# --- Save the structure to files ---

# Save as POSCAR format (commonly used by VASP)
# sort=True ensures consistent atom ordering if the slab is regenerated.
# vasp5=True uses the VASP 5 format (recommended).
print(f"\nSaving structure to POSCAR format: {poscar_filename}")
ase.io.write(poscar_filename, slab, format='vasp', sort=False, vasp6=True)

# Save as XYZ format (simple, human-readable, widely compatible)
print(f"Saving structure to XYZ format:    {xyz_filename}")
ase.io.write(xyz_filename, slab, format='xyz')

# --- Now do the same with right leads -> inverted slab ---
slab.positions=slab.positions * -1
slab.positions = slab.positions + slab.cell.diagonal()

# --- Define output filenames ---
# Create descriptive filenames based on parameters
base_filename = f"{element}_{''.join(map(str, surface_index))}_{size[0]}x{size[1]}x{size[2]}_right"
poscar_filename = base_filename + ".poscar"
xyz_filename = base_filename + ".xyz"

# --- Save the structure to files ---

# Save as POSCAR format (commonly used by VASP)
# sort=True ensures consistent atom ordering if the slab is regenerated.
# vasp5=True uses the VASP 5 format (recommended).
print(f"\nSaving structure to POSCAR format: {poscar_filename}")
ase.io.write(poscar_filename, slab, format='vasp', sort=False, vasp6=True)

# Save as XYZ format (simple, human-readable, widely compatible)
print(f"Saving structure to XYZ format:    {xyz_filename}")
ase.io.write(xyz_filename, slab, format='xyz')

