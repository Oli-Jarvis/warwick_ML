import os
import sys
import time
import subprocess
import torch
import numpy as np
import random

def get_slurm_remaining_time(job_id):
    """
    Queries squeue to get remaining time for a job_id.
    Returns remaining time in seconds.
    Returns infinity if command fails, format is unparseable, or no job_id is provided.
    """
    if not job_id:
        return float('inf')

    try:
        # -h: no header, -j: job id, -o %L: time remaining
        result = subprocess.run(
            ['squeue', '-h', '-j', job_id, '-o', '%L'],
            capture_output=True, text=True
        )
        time_str = result.stdout.strip()

        if not time_str or "UNLIMITED" in time_str:
            return float('inf')

        # Parse formats: "MM:SS", "HH:MM:SS", "D-HH:MM:SS"
        days = 0
        if '-' in time_str:
            parts = time_str.split('-')
            days = int(parts[0])
            time_str = parts[1]

        time_parts = list(map(int, time_str.split(':')))

        seconds = 0
        if len(time_parts) == 3:  # HH:MM:SS
            seconds = time_parts[0] * 3600 + time_parts[1] * 60 + time_parts[2]
        elif len(time_parts) == 2:  # MM:SS
            seconds = time_parts[0] * 60 + time_parts[1]
        else:
            return float('inf')

        return seconds + (days * 86400)

    except Exception:
        return float('inf')


class CheckpointHandler:
    def __init__(self, log_dir, slurm_threshold_sec=3000, slurm_check_interval_sec=2400):
        """
        Initializes checkpoint handler.
        Args:
            log_dir:
            slurm_threshold_sec:
            slurm_check_interval:
        """
        self.log_dir = log_dir
        self.ckpt_path = os.path.join(log_dir, "checkpoint.pt")
        self.exp_path = os.path.join(log_dir, "experience.pt")
        self.hist_path = os.path.join(log_dir, "full_history.pt")

        # Slurm settings
        self.slurm_threshold = slurm_threshold_sec
        self.slurm_interval = slurm_check_interval_sec
        self.last_check_time = time.time()
        self.slurm_job_id = os.environ.get('SLURM_JOB_ID')

    def save(self, step, agent, optimizer, log_z, oracle_handler, experience, full_history, rng_instance):
        """Saves state + All Random Number Generator States."""
        print(f"Saving checkpoint at step {step}...")

        checkpoint_data = {
            'step': step,
            'agent_state_dict': agent.net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'log_z': log_z.data,
            'oracle_calls': oracle_handler.oracle_calls,

            # --- RNG STATES ---
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy_rng': np.random.get_state(),
            'python_rng': random.getstate(),
            'trainer_rng_state': rng_instance.bit_generator.state  # Save the specific local rng
        }

        torch.save(checkpoint_data, self.ckpt_path)
        torch.save(experience, self.exp_path)
        torch.save(full_history, self.hist_path)
        print("Checkpoint saved.")

    def attempt_restart(self, agent, optimizer, log_z, oracle_handler, prior_path, device, rng_instance):
        """Restores state + All Random Number Generator States."""
        if os.path.exists(self.ckpt_path):
            print(f"Resuming training from checkpoint: {self.ckpt_path}")
            checkpoint = torch.load(self.ckpt_path, map_location=device, weights_only=False)

            # load models
            agent.net.load_state_dict(checkpoint['agent_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            log_z.data = checkpoint['log_z']
            oracle_handler.oracle_calls = checkpoint.get('oracle_calls', 0)
            start_step = checkpoint['step'] + 1

            # restore RNG States
            if 'torch_rng' in checkpoint:
                torch.set_rng_state(checkpoint['torch_rng'])
            if 'cuda_rng' in checkpoint and checkpoint['cuda_rng'] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
            if 'numpy_rng' in checkpoint:
                np.random.set_state(checkpoint['numpy_rng'])
            if 'python_rng' in checkpoint:
                random.setstate(checkpoint['python_rng'])
            if 'trainer_rng_state' in checkpoint:
                rng_instance.bit_generator.state = checkpoint['trainer_rng_state']

            # load memory buffers
            experience = torch.load(self.exp_path, weights_only=False) if os.path.exists(self.exp_path) else None
            full_history = torch.load(self.hist_path, weights_only=False) if os.path.exists(self.hist_path) else None

            return start_step, experience, full_history
        else:
            print("No checkpoint found. Starting fresh.")
            weights = torch.load(prior_path, map_location=device, weights_only=False)
            agent.net.load_state_dict(weights)
            return 0, None, None

    def slurm_time_limit_approaching(self):
        """
        Checks if Slurm time limit is near.
        Returns True if remaining time < threshold.
        Returns False if no Slurm job or ample time remains.
        """
        if not self.slurm_job_id:
            return False

        # Only check periodically to save overhead
        if time.time() - self.last_check_time > self.slurm_interval:
            self.last_check_time = time.time()
            remaining = get_slurm_remaining_time(self.slurm_job_id)
            if remaining < self.slurm_threshold:
                print(f"Slurm time limit approaching ({remaining}s left).")
                return True

        return False