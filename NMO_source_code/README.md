# Beyond Drug Discovery: The Nanotechnology Molecular Optimization (NMO) Benchmark

Code for the paper *"Beyond Drug Discovery: The Nanotechnology Molecular Optimization (NMO) Benchmark"* 

NMO is a molecular optimization benchmark suite targeting real quantum physics problems in nanotechnology, replacing simple drug-likeness proxies with semi-empirical quantum simulations. We introduce three oracles (Phonon Transport, Thermoelectrics, Molecular Optomechanics), a novel molecular representation (GGS), and a baseline optimization method based on genetic GFlowNets.

## Structure

| Package | Description |
|---|---|
| [`GGS/`](GGS/README.md) | Graph Group SELFIES — fragment-based molecular representation valid by construction, with native molecule-electrode anchor modeling |
| [`NMO/`](NMO/README.md) | NMO Benchmark Suite — the three quantum physics oracles and evaluation protocol |
| [`genetic_GFN_framework/`](genetic_GFN_framework/README.md) | Baseline method — genetic GFN with GGS encoding, transformer policy, and adaptive stability mechanisms |

All three packages are **independent of each other** and can be installed and used separately. See each package's README for details.

## Setup

To install all packages and dependencies at once, run:

```bash
bash setup.sh
```
