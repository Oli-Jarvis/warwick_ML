# NMO — API Reference

This document covers the three core computation modules of the NMO package:
`electronic_structure`, `electronic_transport`, and `phononic_transport`.
All classes are importable directly from the top-level package.

```python
from NMO import (
    Electronic_Structure_Calculator,
    Electronic_Transport_Calculator_torch,
    Electronic_Transport_Estimator_torch,
    Phononic_Transport_Estimator_torch,
)
```

---

## Table of Contents

1. [Electronic Structure](#1-electronic-structure)
   - [Electronic_Structure_Calculator](#electronic_structure_calculator)
2. [Electronic Transport](#2-electronic-transport)
   - [Electronic_Transport_Calculator](#electronic_transport_calculator)
   - [Electronic_Transport_Calculator_torch](#electronic_transport_calculator_torch)
   - [Electronic_Transport_Estimator_torch](#electronic_transport_estimator_torch)
3. [Phononic Transport](#3-phononic-transport)
   - [Phononic_Transport_Estimator_torch](#phononic_transport_estimator_torch)
4. [Physical Constants](#4-physical-constants)
5. [Utility Functions](#5-utility-functions)
6. [Computation Pipeline](#6-computation-pipeline)

---

## 1. Electronic Structure

### `Electronic_Structure_Calculator`

Wraps the **dxtb** differentiable tight-binding calculator (GFN1-xTB or GFN2-xTB)
and provides a unified batched interface for electronic structure properties.
Geometry optimisation and Hessian calculation delegate to an external **xtb** binary
for numerical stability.

#### Constructor

```python
Electronic_Structure_Calculator(
    numbers,           # torch.Tensor  — atomic numbers, shape (N,) or (B, N)
    positions,         # torch.Tensor  — positions in Bohr, shape (N, 3) or (B, N, 3)
    charge=None,       # torch.Tensor  — total charge; defaults to 0 per molecule
    fixed_atoms=None,  # torch.Tensor  — integer mask: 0 = frozen, 1 = free, shape (N,) or (B, N)
    opts_add=None,     # dict          — overrides for any dxtb calculator option
    add_fake_batch=True,  # bool       — auto-add batch dim when input is unbatched
    gfn=1,             # int           — xTB parametrisation: 1 (GFN1) or 2 (GFN2)
)
```

A single-molecule input (2-D tensors) is silently promoted to a batch of size 1
when `add_fake_batch=True`. GPU is used automatically when CUDA is available.

**Default dxtb options** (overridable via `opts_add`):

| Key | Default | Notes |
|-----|---------|-------|
| `scf_mode` | `SCF_MODE_FULL` | Full SCF (no early exit) |
| `scp_mode` | `SCP_MODE_CHARGE` | Self-consistent charges |
| `damp` | `0.1` | Damping for charge mixing |
| `damp_dynamic` | `True` | Adaptive damping |
| `fermi_etemp` | `300` | Electronic temperature in K |
| `fermi_maxiter` | `200` | Max Fermi-level iterations |
| `verbosity` | `0` | Silent |

#### Factory Class Methods

```python
Electronic_Structure_Calculator.from_mol_batch(mol_list, opts_add=None, gfn=1, dd=None)
```
Create from a list of **RDKit** `Mol` objects. If no 3-D conformer is embedded,
`EmbedMolecule` with `randomSeed=0` is called automatically.

```python
Electronic_Structure_Calculator.from_coord_xyz(coord_xyz_path, opts_add=None, gfn=1, dd=None)
```
Create from a single XYZ file.

```python
Electronic_Structure_Calculator.from_coord_xyz_batch(coord_xyz_path_list, opts_add=None, gfn=1, dd=None)
```
Create from a list of XYZ files. Molecules are zero-padded into a batch tensor via
`tad_mctc.batch.pack`. "X" dummy atoms are silently dropped.

#### Cached Properties

All properties are computed on first access and cached. Units are as stated.

| Property | Shape | Unit | Description |
|----------|-------|------|-------------|
| `energy` | `(B,)` | Ha | Total GFN energy |
| `mo_energies` | `(B, N_ao)` | Ha | Molecular orbital energies |
| `coefficients` | `(B, N_ao, N_ao)` | — | MO expansion coefficients in AO basis |
| `overlap` | `(B, N_ao, N_ao)` | — | Overlap matrix S |
| `occupation` | `(B, N_ao)` | — | MO occupation numbers |
| `charge` | `(B,)` | e | System charge (auto-initialised to 0) |
| `hessian` | `(B, 3N, 3N)` | Ha/Bohr² | Mass-unweighted Hessian (via external xtb) |

> **Note:** Accessing `hessian` triggers `get_hessian_external_xtb()`.
> Accessing any other property triggers a full SCF via dxtb.

#### Methods

```python
calc.optimize_geometry_external(total_cpus=None)
    -> (converged_list: np.ndarray,
        hl_gaps: np.ndarray,
        failure_reason_list: List[str])
```
Runs `xtb --opt extreme --gfn {gfn}` in parallel subprocesses.
Thread count per molecule is determined automatically based on system size
(≤15 atoms → 1 CPU; 16–30 → 2; 31–55 → 4; >55 → 8).
Updates `self.numbers` and `self.positions` in-place for converged molecules.

- **Convergence checks:** Output files (`xtbopt.xyz`, `.xtboptok`, `xtbtopo.mol`) must
  be present; the optimised structure must be a single fragment; Au atoms must have
  degree ≤ 1; HOMO–LUMO gap must exceed `HL_GAP_THRESH = 0.2 eV`.
- **Returns:** Boolean array `converged_list`, gap array `hl_gaps` (−1 if failed),
  and per-molecule failure descriptions.

```python
calc.get_hessian_external_xtb(total_cpus=None)
    -> (hessians: torch.Tensor,   # shape (B, 3N, 3N), Ha/Bohr²
        failure_reasons: List[str])
```
Runs `xtb --hess --gfn {gfn}` in parallel. Zero matrix returned for failed molecules.
Uses `/dev/shm` as scratch directory when available.

```python
calc.terahertz_upconversion_external(total_cpus=None) -> np.ndarray  # shape (B,)
```
Calls the PTB-enabled xtb binary (`$XTB_PTB_BIN`) with `--ptb --hess --raman 289.15 785`.
Parses `THz target P:` from stdout and returns `10**P` per molecule.

```python
calc(positions_ang) -> (energy_np, gradients_np)
```
ASE-compatible `__call__` interface (positions in Å, returns forces in Ha/Bohr).
Supports fixed-atom masking via `self.fixed_atoms`.

#### Timeouts and Error Handling

- External xtb processes time out after **25 minutes** (`TIMEOUT = 25 * 60`).
- Failed molecules are assigned zero tensors/arrays; failure messages are propagated
  through `failure_reason_list`.

---

## 2. Electronic Transport

All transport classes implement NEGF (Non-Equilibrium Green's Function) Landauer
transport theory. The thermoelectric transport coefficients follow:

> Bürkle et al., *Phys. Rev. B* **91**, 165419 (2015), Eqs. (3)–(6).

The Fermi-level alignment shift follows:

> Verzijl & Thijssen, *J. Phys. Chem. C* **116**, 24393 (2012).

---

### `Electronic_Transport_Calculator`

Reference NumPy implementation for a single molecule with pre-built ECC matrices.

#### Constructor

```python
Electronic_Transport_Calculator(
    H_ECC,               # np.ndarray  — extended-central-cell Hamiltonian (Ha), shape (N_orb, N_orb)
    S_ECC,               # np.ndarray  — extended-central-cell overlap, shape (N_orb, N_orb)
    E_min,               # float       — lower energy bound in eV
    E_max,               # float       — upper energy bound in eV
    N_E_points,          # int         — number of energy grid points
    g_surf_left_path,    # str         — directory containing left-lead data
    g_surf_right_path,   # str         — directory containing right-lead data
    coord_xyz_ecc,       # array-like  — ECC atomic coordinates (for geometry validation)
    T=300,               # float       — temperature in K
    WBL=False,           # bool        — use Wide-Band-Limit approximation
    strict=False,        # bool        — validate lead–ECC atom alignment on init
)
```

**Lead data directory** must contain:
- `H_00.dat`, `S_00.dat` — bulk principal-layer matrices (complex, space-separated)
- `G_surf` — surface Green's function binary (interleaved real/imag float64)
- `dos_surf.dat` — energy points for `G_surf` (row 0)
- `detailed_first.out` — DFTB+ output; Fermi level parsed from `Fermi level ... eV` line

#### Fermi-Level Alignment

On construction (or first property access), a diagonal shift δ is applied:

```
δ = mean_k [ (H_00_bulk[k,k] − H_ECC_lead[k,k]) / S_ECC_lead[k,k] ]
```

averaged over left and right leads. `H_ECC` and its partitions are shifted in-place:
`H_ECC ← H_ECC + δ · S_ECC`.

#### Cached Properties

| Property | Unit | Description |
|----------|------|-------------|
| `E` | Ha | Energy grid, shape `(N_E_points,)` |
| `E_fermi` | eV | Fermi energy (average of left/right leads) |
| `N_orb_lead` | — | Total number of lead AOs (2 × principal layer) |
| `tau_el` | — | Transmission function T(E), shape `(N_E_points,)` |
| `K_0` / `G_el` | G₀ | Electrical conductance at E_fermi |
| `K_1` | eV·G₀ | First transport moment |
| `K_2` | eV²·G₀ | Second transport moment |
| `S_el` | μV/K | Seebeck coefficient (thermopower) |
| `kappa_el` | pW/K | Electronic thermal conductance |

#### Methods

```python
calc.calculate_tau_el() -> np.ndarray   # shape (N_E_points,)
```
Full NEGF: computes self-energies `Σ_L/R(E) = V†_{CX} g_surf(E) V_{CX}`, broadening
functions `Γ_{L/R} = −2 Im Σ_{L/R}`, and transmission
`T(E) = Tr[Γ_L G^r Γ_R G^a]`. Regularisation parameter η = 10⁻⁶.

```python
calc.calculate_K_n(n, mu=-1) -> float
```
Computes `K_n = ∫ T(E) (−∂f/∂E) (E − μ)^n dE` via trapezoidal integration.
`mu=-1` uses the Fermi energy from lead files.

```python
calc.calculate_G_el(mu=-1) -> float   # G₀
calc.calculate_S_el(mu=-1) -> float   # μV/K
calc.calculate_kappa_el(mu=-1) -> float  # pW/K
```
Wrappers around `K_n`:
- `G_el = K_0`
- `S_el = −K_1 / (e T K_0)` [in μV/K]
- `κ_el = (2/hT)(K_2 − K_1²/K_0)` [in pW/K]

```python
calc.check_system()
```
Validates that ECC atom positions match POSCAR files in the lead directories
(up to a global rigid shift). Raises `ValueError` on mismatch.

---

### `Electronic_Transport_Calculator_torch`

GPU-accelerated batched version. Inherits from `Electronic_Transport_Calculator`.
Takes an `Electronic_Structure_Calculator` directly and reconstructs H and S in the
AO basis from MO energies and coefficients.

#### Constructor

```python
Electronic_Transport_Calculator_torch(
    el_structure_calculator,   # Electronic_Structure_Calculator
    E_min,                     # float  — lower energy bound in eV
    E_max,                     # float  — upper energy bound in eV
    N_E_points,                # int    — number of energy grid points
    g_surf_left_path,          # str    — left-lead directory
    g_surf_right_path,         # str    — right-lead directory
    T=300,                     # float  — temperature in K
    WBL=False,                 # bool   — Wide-Band-Limit
)
```

The full Hamiltonian is reconstructed as:
```
H = S C ε C^T S
```
where `C` are MO coefficients, `ε` are MO energies (diagonal), and `S` is the
overlap matrix — all taken from `el_structure_calculator`.

**Key differences from the NumPy base class:**

- All matrix operations use `torch.bmm`/`einsum`, batched over molecules.
- Surface Green's functions are loaded as `torch.Tensor` (complex64).
- The energy grid is a `torch.Tensor` on the same device as H/S.
- Fermi-level shift is applied per-batch-element.

The `tau_el` property returns a `torch.Tensor` of shape `(B, N_E_points)`.
The scalar transport coefficients (`G_el`, `S_el`, `kappa_el`) return
`torch.Tensor` of shape `(B,)`.

---

### `Electronic_Transport_Estimator_torch`

Fast estimator using the **Wide-Band-Limit** (WBL) only. Requires no lead surface
Green's function files — uses only the central-molecule AO block. Inherits from
`Electronic_Transport_Calculator`.

#### Constructor

```python
Electronic_Transport_Estimator_torch(
    el_structure_calculator,   # Electronic_Structure_Calculator
    E_min,                     # float  — lower energy bound in eV
    E_max,                     # float  — upper energy bound in eV
    N_E_points,                # int    — number of energy grid points
    anchor_atom=79,            # int    — atomic number of anchor atom (default: Au)
    T=300,                     # float  — temperature in K
)
```

**Anchor detection:** Exactly two atoms with `numbers == anchor_atom` must be present
per molecule. The first is taken as the left contact, the second as the right contact.
AO ranges for each anchor are computed from the GFN1 AO count table (`gfn1_ao_num_by_ao_num`).

**WBL self-energy:** `Σ_{L/R}(E) = i π · 0.036 · Ha2eV · I` (diagonal, imaginary).

The `E_fermi` property reads the Fermi energy from MO occupations (midpoint of
HOMO–LUMO gap) rather than from lead files.

The `tau_el` and scalar transport coefficient properties have the same shapes and
units as `Electronic_Transport_Calculator_torch`.

---

## 3. Phononic Transport

### `Phononic_Transport_Estimator_torch`

Calculates phonon heat transport through a single-molecule junction using the
Landauer formula with a **Debye model** for the electrodes.

> Markussen, *J. Chem. Phys.* **139**, 244309 (2013).  
> Mingo, *Phys. Rev. B* **74**, 125402 (2006).

#### Constructor

```python
Phononic_Transport_Estimator_torch(
    el_structure_calculator,  # Electronic_Structure_Calculator  — provides the Hessian
    E_D,                      # float  — Debye cutoff energy in meV
    N_E_points,               # int    — number of frequency grid points
    gamma=-7,                 # float  — molecule–electrode force constant in eV/Å²
    anchor_atom=79,           # int    — atomic number of anchor atom (default: Au)
    T=300,                    # float  — temperature in K
    hessian=None,             # torch.Tensor or None — pre-computed Hessian (overrides calculator)
)
```

If `hessian` is not provided, `el_structure_calculator.hessian` is used (triggering
an external xtb Hessian calculation if not yet cached).

**Anchor detection:** Same two-anchor convention as `Electronic_Transport_Estimator_torch`.

**`gamma` convention:** The coupling force constant between molecule and electrode in
eV/Å², internally converted to atomic units (Ha/Bohr²) as:
`γ_hb = gamma · (eV2Ha / ang2bohr²)`.

#### Cached Properties

| Property | Shape | Unit | Description |
|----------|-------|------|-------------|
| `w` | `(N_E_points,)` | 1/s | Frequency grid (0 to 1.1 · ω_D) |
| `E` | `(N_E_points,)` | meV | Energy grid corresponding to `w` |
| `w_D` | scalar | 1/s | Debye cutoff frequency |
| `g0` | `(N_E_points,)` | complex | Bare electrode surface Green's function (Debye model) |
| `sigma` | `(N_E_points,)` | complex | Coupling self-energy Σ(ω) |
| `tau_ph` | `(B, N_E_points,)` | — | Phonon transmission function T(ω) |
| `kappa_ph` | `(B,)` | pW/K | Phononic heat conductance |

#### Electrode Surface Green's Function

The imaginary part of the Debye surface Green's function is:
```
Im g₀(ω) = −π · (3ω) / (2ω_D³)   for ω ≤ ω_D
Im g₀(ω) = 0                       for ω > ω_D
```
The real part is obtained via Kramers–Kronig (Hilbert transform, `scipy.signal.hilbert`).

The effective coupling self-energy with mass renormalisation:
```
γ' = γ / √(M_L · M_C)
Σ(ω) = γ'² · g₀(ω) / (1 + γ' · g₀(ω))
```

#### Methods

```python
@torch.no_grad()
calc.calculate_tau_ph() -> torch.Tensor   # shape (B, N_E_points)
```
Computes per-molecule phonon transmission via NEGF:
1. Builds mass-weighted Hessian block `K_CC` (in Ha/Bohr² · amu units).
2. Applies momentum conservation: diagonal corrected by `K_CC.sum(axis=-1)`.
3. For each molecule: `T(ω) = Tr[Γ_L G^r Γ_R G^a]` where self-energies are
   placed at the anchor-atom diagonal positions.
4. Zero-padded rows/columns (padding atoms) are detected via zero diagonal
   entries and excluded from the NEGF calculation.

```python
@torch.no_grad()
calc.calculate_kappa_ph() -> torch.Tensor   # shape (B,)
```
Integrates:
```
κ_ph = (ℏ / 2π kT²) ∫ T(ω) · ω² · n_BE(ω) · [n_BE(ω) + 1] dω
```
where `n_BE(ω)` is the Bose–Einstein distribution, via trapezoidal integration.
Result in pW/K.

---

## 4. Physical Constants

Exported from `NMO.constants`:

```python
from NMO import __Ha2eV__, __hP__, __e0__, __G0__
```

| Symbol | Value | Unit |
|--------|-------|------|
| `__Ha2eV__` | 27.21138602 | eV/Ha |
| `__eV2Ha__` | 0.0367493 | Ha/eV |
| `__ang2bohr__` | 1.88973 | Bohr/Å |
| `__hP__` | 4.135667662 × 10⁻¹⁵ | eV·s |
| `__e0__` | 1.6021766208 × 10⁻¹⁹ | C |
| `__G0__` | 2e²/h | G₀ (conductance quantum) |
| `__k_B__` | 3.167 × 10⁻⁶ | Ha/K |
| `__h__` | 1.51983 × 10⁻¹⁶ | Ha·s |

---

## 5. Utility Functions

```python
from NMO import interpolate_energy_dependent_complex_matrix, print_vram_usage
```

```python
interpolate_energy_dependent_complex_matrix(
    precalculated_energy_points,  # np.ndarray  — shape (N_E_old,), in Ha
    matrix,                       # np.ndarray  — shape (N_E_old, d, d), complex
    new_energy,                   # np.ndarray  — shape (N_E_new,), in Ha
) -> np.ndarray   # shape (N_E_new, d, d)
```
PCHIP monotone cubic spline interpolation of an energy-dependent complex matrix,
applied independently to real and imaginary parts.

```python
print_vram_usage()
```
Prints total / free / used GPU VRAM in MB. No-op on CPU-only systems.

Additional utilities (importable from `NMO`):

| Function | Description |
|----------|-------------|
| `align_molecule(positions, numbers, target_axis, ...)` | Rotate molecule batch to align junction axis using Rodrigues rotation |
| `add_gold(numbers_C, positions_C, ...)` | Attach pre-computed Au tip geometry to centre molecule |
| `find_anchor_atom_indices(numbers_tensor, anchor_atom=16)` | Locate left/right anchor atom indices (validates exactly 2 per molecule) |
| `read_kpoints_weights(path)` | Read k-point weights from lead calculation directory |

---

## 6. Computation Pipeline

The typical sequence for evaluating a molecule batch:

```
1. Electronic_Structure_Calculator
       ├── .optimize_geometry_external()     ← GFN1/2-xTB geometry relaxation
       ├── .energy / .mo_energies / .coefficients / .overlap
       │      (dxtb SCF, GPU-accelerated)
       └── .get_hessian_external_xtb()       ← required for phononic transport

2a. Electronic_Transport_Calculator_torch    ← full NEGF with lead data
      └── .G_el / .S_el / .kappa_el

2b. Electronic_Transport_Estimator_torch     ← WBL-only, no lead files needed
      └── .G_el / .S_el / .kappa_el

3.  Phononic_Transport_Estimator_torch       ← Debye-model NEGF
      └── .kappa_ph
```

For high-throughput screening, use `Oracle_Handler_Smiles` or `Oracle_Handler_GGS`
(see main [README](README.md)), which orchestrate steps 1–3 automatically, handle
parallelism, and write results to HDF5.
