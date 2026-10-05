# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys

# Running from the pmo/ directory: add pmo/ so that 'main.*' imports resolve,
# and add the repo root so that 'scripts.exps.pmo.*' imports in existing code resolve.
_PMO_DIR = os.path.dirname(os.path.realpath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_PMO_DIR)))
sys.path.insert(0, _PMO_DIR)
sys.path.insert(0, _REPO_ROOT)

import argparse
import yaml
from time import time
from NMO import Oracle_Handler_Smiles
from main.genmol.run_nmo import NMO_Optimizer


def main():
    start_time = time()
    parser = argparse.ArgumentParser(description='GenMol optimization on the NMO benchmark')
    parser.add_argument('--nmo_config', required=True,
                        help='Path to the NMO .ini config file (oracle settings, fitness_func, etc.)')
    parser.add_argument('-c', '--config', default='hparams.yaml',
                        help='Path to GenMol hparams YAML file')
    parser.add_argument('--task_name', default='nmo',
                        help='Label used for result files (default: nmo)')
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--max_oracle_calls', type=int, default=10000,
                        help='Max oracle calls tracked by GenMol (NMO also has its own limit in .ini)')
    parser.add_argument('--freq_log', type=int, default=100)
    parser.add_argument('-s', '--seed', type=int, default=1)
    args = parser.parse_args()

    path_results = os.path.join(_PMO_DIR, 'main/genmol/results')

    if args.output_dir is None:
        args.output_dir = path_results

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    try:
        config = yaml.safe_load(open(args.config))
    except FileNotFoundError:
        config = yaml.safe_load(open(os.path.join(_PMO_DIR, 'main/genmol', args.config)))

    print(f'NMO config: {args.nmo_config}')
    print(f'Task name:  {args.task_name}')
    print(f'Seed:       {args.seed}')

    oracle = Oracle_Handler_Smiles(args.nmo_config)
    optimizer = NMO_Optimizer(args=args)
    optimizer.optimize(oracle=oracle, config=config, seed=args.seed)

    hours = (time() - start_time) / 3600.0
    print(f'---- The whole process takes {hours:.2f} hours ----')


if __name__ == '__main__':
    main()
