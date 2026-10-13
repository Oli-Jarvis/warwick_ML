---
license: cc-by-4.0
---

This dataset comprises the candidates identified by the baseline for all three design tasks of the NMO task. It contains the following information:

| Column | Type | Description |
|---|---|---|
| `task` | string | `phonon`, `thermoelectric`, or `optomechanics` |
| `hash` | string | Unique molecule hash |
| `benchmark_context__model_name` | string | Optimizer name |
| `benchmark_context__model_version` | string | Optimizer version |
| `benchmark_context__model_url` | string | Repository or paper URL |
| `benchmark_context__hyperparameters` | string | Hyperparameter YAML |
| `benchmark_context__grammar` | string | Grammar/vocabulary file (if applicable) |
| `calculation_settings__geometry_method` | string | Geometry optimization method |
| `calculation_settings__temperature` | float | Simulation temperature [K] |
| `calculation_settings__oracle_config` | string | Full oracle config file(s) |
| `molecular_junction__smiles` | string | SMILES with Au-S anchors |
| `molecular_junction__sa_score` | float | Synthetic accessibility score |
| `molecular_junction__hl_gap` | float | HOMO-LUMO gap |
| `molecular_junction__oracle_call` | int | Oracle call index |
| `structure__species` | list[string] | Element symbols of all atoms |
| `structure__positions` | list[list[float]] | 3D coordinates [Å] |
| `phonon_transport__kappa` | float | Phonon thermal conductance κ_ph [pW/K] |
| `phonon_transport__energy` | list[float] | Phonon energy grid [eV] |
| `phonon_transport__transmission` | list[float] | Phonon transmission spectrum |
| `electronic_transport__conductance` | float | Electrical conductance G [G₀] |
| `electronic_transport__seebeck` | float | Seebeck coefficient S [μV/K] |
| `electronic_transport__kappa_el` | float | Electronic thermal conductance [pW/K] |
| `electronic_transport__zt` | float | Figure of merit ZT at 300 K |
| `electronic_transport__energy` | list[float] | Electronic energy grid [eV] |
| `electronic_transport__transmission` | list[float] | Electronic transmission spectrum |
| `upconversion__log_p_upconversion` | float | log₁₀(P_upconversion) |
| `upconversion__log_p_upconversion_scaled` | float | Length-and-area-scaled log₁₀(P) |
| `upconversion__molecular_length` | float | Molecular length [Å] |
| `upconversion__surface_area` | float | Projected surface area [Å²] |

Fields that do not apply to a given task are stored as `null`.