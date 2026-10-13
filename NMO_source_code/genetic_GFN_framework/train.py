import random
import sys, os
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import configparser
import numpy as np
import subprocess
from rdkit import Chem
from group_selfies import GroupGrammar

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    #print git information
    try:
        # Run the 'git rev-parse HEAD' command
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=current_dir,
            capture_output=True,
            text=True,
            check=True
        )

        commit_hash = result.stdout.strip()

        print(f"Current commit hash: {commit_hash}")

        result_short = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=current_dir,
            capture_output=True,
            text=True,
            check=True
        )
        short_hash = result_short.stdout.strip()
        print(f"Short commit hash: {short_hash}")

    except subprocess.CalledProcessError as e:
        print(f"Error running Git command in {current_dir}:")
        print(e.stderr)
    except FileNotFoundError:
        print("Git command not found. Make sure Git is installed and in your PATH.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train GFlow_Mol')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, help='seed', default=-1)
    parser.add_argument("config", type=str, help='Path to config file')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    seed = args.seed
    if seed != -1:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed=seed)
        #torch.backends.cudnn.deterministic = True

from utils import Experience, unique, padding_and_valid_mask, calc_descriptors
from GGS import validate_molecule
from action_space import Action_Space_GroupSelfies, Action_Space_Smiles
from model import get_model
from NMO import Oracle_Handler_GGS, Oracle_Handler_Smiles
from genetic_search import GeneticOperatorHandler_Smiles, GeneticOperatorHandler_GroupSelfies
from analysis import analyze_final_history, plot_full_history
from train_utils import CheckpointHandler


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GFlow_Mol_Trainer():

    def __init__(self, config_file, seed = -1):
        self.config = configparser.ConfigParser()
        self.config.read(config_file, encoding='utf-8')

        self.seed = seed
        self.config_file = config_file

        self.grammar_path = self.config.get("General", "grammar_path")
        self.max_seq_length = self.config.getint("General", "max_seq_length")
        self.encoding_type = self.config.get("General", "encoding_type")
        #this error handling is to make backward compatible with config files where rank_coefficient was part of the [General] section.
        try:
            self.rank_coefficient = self.config.getfloat("Training", "rank_coefficient")
        except configparser.NoOptionError:
            print("Deprecation warning: rank_coefficient not found in [Training], falling back to [General]. Please move it to [Training].")
            self.rank_coefficient = self.config.getfloat("General", "rank_coefficient", fallback=0.01)
        self.num_layers = self.config.getint("General", "num_layers")
        self.d_model = self.config.getint("General", "d_model")
        self.model = self.config.get("General", "model")

        self.prior_path = self.config.get("Training", "prior_path")
        self.log_dir = self.config.get("Training", "log_dir")
        os.makedirs(self.log_dir, exist_ok=True)
        self.batch_size = self.config.getint("Training", "batch_size")
        self.n_steps = self.config.getint("Training", "n_steps", fallback = 10000)

        self.learning_rate = self.config.getfloat("Training", "learning_rate")
        self.learning_rate_z = self.config.getfloat("Training", "learning_rate_z")
        self.oracle_calculated_props = self.config.get("Oracle", "calculated_props")
        self.n_oracle_processes = self.config.getint("Training", "n_oracle_processes", fallback = 8)
        self.use_masking = self.config.getboolean("Training", "use_masking", fallback = True)
        self.use_max_seq_length_padding = self.config.getboolean("Training", "use_max_seq_length_padding", fallback = True)
        self.temperature = self.config.getfloat("Training", "temperature", fallback = 1.0)
        self.gradient_clipping = self.config.getfloat("Training", "gradient_clipping", fallback=float('inf'))
        self.dynamic_explor_exploit = self.config.getboolean("Training", "dynamic_explor_exploit", fallback=False)
        self.debug = self.config.getboolean("Training", "debug", fallback = False)
        self.dynamic_cooldown = self.config.getboolean("Training", "dynamic_cooldown", fallback = False)
        self.use_SMARTS_filters = self.config.getboolean('Training', 'use_SMARTS_filters', fallback=True)

        self.memory_size = self.config.getint("Replay Training", "memory_size")
        self.experience_replay = self.config.getint("Replay Training", "experience_replay")
        self.n_experience_iterations = self.config.getint("Replay Training", "n_experience_iterations")
        self.beta = self.config.getfloat("Replay Training", "beta")
        self.kl_coefficient = self.config.getfloat("Replay Training", "kl_coefficient")
        self.descriptor_weight = self.config.getfloat("Replay Training", "descriptor_weight", fallback = 0.0)
        self.strict_memory_handling = self.config.getboolean("Replay Training", "strict_memory_handling", fallback = True)

        self.genetic_search = self.config.getboolean("Genetic Search", "genetic_search")
        self.population_size = self.config.getint("Genetic Search", "population_size")
        self.ga_generations = self.config.getint("Genetic Search", "ga_generations")
        self.mutation_rate = self.config.getfloat("Genetic Search", "mutation_rate")
        self.offspring_size = self.config.getint("Genetic Search", "offspring_size")

        print("--------- Configuration: ----------")
        for section in self.config.sections():
            for key, value in self.config.items(section):
                print(f"{section}.{key} = {value}")
        print("-----------------------------------")



        self.rng = rng if seed != -1 else np.random.default_rng()

        if self.encoding_type == "GGS":
            self.action_space = Action_Space_GroupSelfies.from_grammar_path(self.grammar_path)
            self.ga_handler = GeneticOperatorHandler_GroupSelfies(self.config, self.action_space, self.seed)
            self.oracle_handler = Oracle_Handler_GGS(self.config_file, self.rng)
            self.offspring_size = int(self.offspring_size/2)  #crossover creates two children
            self.grammar = GroupGrammar.from_file(self.grammar_path)
        elif self.encoding_type == "Smiles":
            self.action_space = Action_Space_Smiles(self.grammar_path)
            self.ga_handler = GeneticOperatorHandler_Smiles(self.config, self.seed)
            self.oracle_handler = Oracle_Handler_Smiles(self.config_file, self.rng)


    sample_stats = []
    dynamic_explore_exploit_active = False
    def train(self):

        #Initialize
        Prior = get_model(self.model, self.action_space, self.batch_size, self.max_seq_length, num_layers=self.num_layers, d_model=self.d_model)
        Prior.to(device)
        Agent = get_model(self.model, self.action_space, self.batch_size, self.max_seq_length, num_layers=self.num_layers, d_model=self.d_model)
        Agent.to(device)

        experience = Experience(self.action_space , self.memory_size, strict = self.strict_memory_handling, seed = self.seed)
        experience_dir = f"{self.log_dir}/experience_memory"
        if not os.path.exists(experience_dir):
            os.makedirs(experience_dir)
        full_history = Experience(self.action_space , -1, strict = False, seed = self.seed)

        print(f"Loading model from {self.prior_path}...")

        if torch.cuda.is_available():
            state_dict = torch.load(self.prior_path, weights_only=False)
        else:
            state_dict = torch.load(self.prior_path, map_location=torch.device('cpu'), weights_only=False)

        # register buffers
        if 'desc_mean' in state_dict:
            Prior.net.register_buffer('desc_mean', state_dict['desc_mean'])
            Prior.net.register_buffer('desc_std', state_dict['desc_std'])

            Agent.net.register_buffer('desc_mean', state_dict['desc_mean'])
            Agent.net.register_buffer('desc_std', state_dict['desc_std'])

            print("Registered descriptor statistics to Prior and Agent.")

        Prior.net.load_state_dict(state_dict, strict=False)
        Agent.net.load_state_dict(state_dict, strict=False)

        #prior remains unchanged
        for param in Prior.net.parameters():
            param.requires_grad = False

        desc_mean = getattr(Prior.net, 'desc_mean', None)
        desc_std = getattr(Prior.net, 'desc_std', None)

        if desc_mean is not None:
            print(f"Loaded descriptor statistics: Mean={desc_mean.mean().item():.2f}, Std={desc_std.mean().item():.2f}")
        else:
            print("Warning: No descriptor statistics found in Prior model. Descriptor loss will be skipped.")

        # set up partition function Z -> here log_z
        if torch.cuda.is_available():
            log_z = torch.nn.Parameter(torch.tensor([5.], device = 'cuda:0'))
        else:
            log_z = torch.nn.Parameter(torch.tensor([5.]))

        # set up optimizer
        optimizer = torch.optim.Adam([{'params': Agent.net.parameters(), 'lr': self.learning_rate},
                                      {'params' : log_z, 'lr' : self.learning_rate_z}])

        ckpt_handler = CheckpointHandler(self.log_dir)
        #attemp restart
        start_step, loaded_exp, loaded_hist = ckpt_handler.attempt_restart(
            Agent, optimizer, log_z, self.oracle_handler, self.prior_path, device, self.rng
        )
        if loaded_exp: experience = loaded_exp
        if loaded_hist: full_history = loaded_hist

        # Load Prior weights (Prior is static, so we always load it from the original path)
        if torch.cuda.is_available():
            Prior.net.load_state_dict(torch.load(self.prior_path, weights_only=False))
        else:
            Prior.net.load_state_dict(torch.load(self.prior_path, map_location='cpu', weights_only=False))

        for param in Prior.net.parameters():
            param.requires_grad = False

        #checkpointing after a certain number of oracle calls
        save_interval = 1000
        current_calls = self.oracle_handler.oracle_calls
        next_save_threshold = ((current_calls // save_interval) + 1) * save_interval

        if self.oracle_handler.oracle_calls >= self.oracle_handler.max_oracle_calls:
            print(f"Start-up Notice: Oracle calls ({self.oracle_handler.oracle_calls}) have already reached the limit .")
            print("Exiting script ...")
            sys.exit(0)

        oracle_calls_exceeded = False
        cooldown_active = False
        dynamic_explore_exploit_active = False
        for step in tqdm(range(start_step, self.n_steps)):

            #dynamic adjustment of rank coefficient
            #if self.dynamic_explor_exploit:
            #    self.rank_coefficient = self.config.getfloat("General", "rank_coefficient")
            #    self.temperature = self.config.getfloat("Training", "temperature", fallback = 1.0)
            #    max_oracle_calls = full_history.get_max_oracle_calls()
            #    lifetime_best_candidate = experience.get_lifetime_best_candidate(max_oracle_calls)
            #    print(f"Lifetime of best candidate {lifetime_best_candidate}")
            #    exploration_patience = self.config.getfloat("Training", "exploration_patience")
            #    if lifetime_best_candidate > exploration_patience * self.batch_size:
            #        self.rank_coefficient = self.config.getfloat("Training", "exploration_rank_coeff")
            #        self.temperature = self.config.getfloat("Training", "exploration_temp")
            #        print(f"Going to exploration mode in {step=} {self.rank_coefficient=} {self.temperature=}")
            #    else:
            #        print(f"Exploitation mode", lifetime_best_candidate)

            sequences, valid_mask, agent_likelihood, entropy = Agent.sample_sequences(self.batch_size, temperature=self.temperature, use_masking = self.use_masking, max_seq_length_padding = self.use_max_seq_length_padding)

            # prevent memory leak
            sequences = sequences.detach()
            agent_likelihood = agent_likelihood.detach()
            entropy = entropy.detach()

            if self.use_max_seq_length_padding:
                assert sequences.shape == (self.batch_size, self.max_seq_length), f"Sequences shape {sequences.shape} does not match expected shape {(self.batch_size, self.max_seq_length)}"

            inital_n_sample = sequences.shape[0]
            # Remove duplicates, and consider only unique seqs
            unique_idxs = unique(sequences)
            sequences = sequences[unique_idxs]
            valid_mask = valid_mask[unique_idxs]
            agent_likelihood = agent_likelihood[unique_idxs]
            entropy = entropy[unique_idxs]

            after_filter_n_sample = sequences.shape[0]
            n_duplicates = inital_n_sample - after_filter_n_sample
            print(f"Step {step}: Removed {inital_n_sample - after_filter_n_sample} duplicate sequences, {after_filter_n_sample} unique sequences remain for oracle evaluation")


            #get prior likelihood
            prior_likelihood, _, _, __ = Prior.sequence_likelihood_for_pretraining(sequences, valid_mask = valid_mask, use_masking = self.use_masking)

            #create group selfies and get sequences to cpu
            # -> stick to squences here ?
            valid_end_token = 0
            invalid_end_token = 0
            encodings = []
            if torch.cuda.is_available():
                sequences_for_eval = sequences.cpu().numpy()
            else:
                sequences_for_eval = sequences.numpy()
            for seq in sequences_for_eval:
                if self.action_space.has_end_token(seq):
                    valid_end_token +=1
                    encodings.append(self.action_space.action_sequence_to_encoding(seq))
                else:
                    encodings.append("")
                    invalid_end_token +=1
            print("Sampled sequences with valid end token: ", valid_end_token, "/", len(sequences_for_eval), "-> Warning: this does not mean that encoding is correct")


            #apply SMARTS filters
            SMARTS_filtered_encodings = []
            SMARTS_filtered_sequences = []
            SMARTS_filters_failed_counter = 0
            invalid_counter_smiles = 0
            filtered_encodings = []
            filtered_smiles = []
            drop_reasons = []
            for i, encoding in enumerate(encodings):

                # Skip empty encodings -> invalid end tokens
                if not encoding or encoding == "":
                    continue

                if self.encoding_type == "Smiles":
                    keep, reason = validate_molecule(encoding)
                elif self.encoding_type == "GGS":
                    try:
                        mol_decoded = self.grammar.decoder(encoding)
                        smiles_decoded = Chem.MolToSmiles(mol_decoded)
                        keep, reason = validate_molecule(smiles_decoded)
                    except ValueError:
                        keep = False
                        reason = ["Invalid SMILES"]
                if not self.use_SMARTS_filters:
                    keep = True
                if keep:
                    SMARTS_filtered_encodings.append(encoding)
                    SMARTS_filtered_sequences.append(sequences_for_eval[i])
                else:
                    if "Invalid SMILES" in reason:
                        invalid_counter_smiles += 1
                    else:
                        SMARTS_filters_failed_counter +=1
                    drop_reasons.append(reason)
                    filtered_encodings.append(encoding)
            print(f"Removed {SMARTS_filters_failed_counter=} encodings/sequences by SMARTS filters. This leaves {len(SMARTS_filtered_encodings)} encodings/sequences for oracle evaluation.")
            print(f"Removed {invalid_counter_smiles=} encodings/sequences because they where invalid")
            print(f"drop_reasons: {drop_reasons}")

            encodings = SMARTS_filtered_encodings
            sequences_for_eval = SMARTS_filtered_sequences

            #sample statistic
            n_valid_after_all_filters = len(sequences_for_eval)

            valid_percentage = n_valid_after_all_filters / inital_n_sample
            invalid_percentage = (invalid_counter_smiles+invalid_end_token)/inital_n_sample
            filtered_percentage = (SMARTS_filters_failed_counter) / inital_n_sample
            duplicates_percentage = (n_duplicates) / inital_n_sample

            total_accounted = (n_valid_after_all_filters + invalid_counter_smiles +
                               invalid_end_token + SMARTS_filters_failed_counter + n_duplicates)
            if total_accounted != inital_n_sample:
                print(f"WARNING: Accounting error! {total_accounted} != {inital_n_sample}")

            # dynamic cooldown for broken grammar -> avoid catastrophic forgetting
            cooldown_activation_threshold = 0.35  # invalid rate exceeds
            cooldown_recovery_threshold = 0.1  # Stop if invalid rate recovers to
            if self.dynamic_cooldown and not cooldown_active and invalid_percentage > cooldown_activation_threshold:
                print(f"activating dynamic cooldown in step {step=}")
                cooldown_active = True
                #self.n_experience_iterations = 16
            elif self.dynamic_cooldown and cooldown_active and invalid_percentage < cooldown_recovery_threshold:
                cooldown_active = False
                #self.n_experience_iterations = self.config.getint("Replay Training", "n_experience_iterations")
            elif self.dynamic_cooldown and cooldown_active:
                print(f"dynamic cooldown active in step {step=}")

            #dynamic explore/exploit
            dynamic_explore_exploit_threshold = 0.70 #once duplicates exceed -> exploration
            dynamic_explore_exploit_reset_threshold = 0.30 #once duplicates go below -> reset rank-based
            if self.dynamic_explor_exploit:
                if not dynamic_explore_exploit_active and duplicates_percentage > dynamic_explore_exploit_threshold:
                    dynamic_explore_exploit_active = True
                    exploration_rank_coeff = self.config.getfloat("Training", "exploration_rank_coeff")
                    if exploration_rank_coeff < self.rank_coefficient:
                        raise ValueError("exploration mode is even more conservative. Aborting")
                    self.rank_coefficient = exploration_rank_coeff
                    print(f"Going to exploration mode in {step=} {self.rank_coefficient=}")
                elif dynamic_explore_exploit_active and duplicates_percentage < dynamic_explore_exploit_reset_threshold:
                    dynamic_explore_exploit_active = False
                    try:
                        self.rank_coefficient = self.config.getfloat("Training", "rank_coefficient")
                    except configparser.NoOptionError:
                        self.rank_coefficient = self.config.getfloat("General", "rank_coefficient", fallback=0.01)
                    print(f"Resetting to normal rank-based sampling in {step=} {self.rank_coefficient=}")


            if self.debug:
                try:
                    tmp_path = os.path.join(self.log_dir, "sample_stats.txt")
                    #write header if file does not exist
                    if not os.path.exists(tmp_path):
                        with open(tmp_path, 'w') as f:
                            f.write("step, inital_n_sample, valid_percentage, invalid_percentage, filtered_percentage, duplicates_percentage, cooldown_active, dynamic_explore_exploit_active\n")
                    with open(tmp_path, 'a') as f:
                        f.write(f"{step}, {inital_n_sample}, {valid_percentage}, {invalid_percentage}, {filtered_percentage}, {duplicates_percentage}, {cooldown_active}, {dynamic_explore_exploit_active}\n")
                except IOError as e:
                    pass


            #Calculate reward but save oracle calls for trajectories which has been already calculated.
            #The calculated ones do not need to be handled as we train from the experience memory and they are
            #already in the experience memory.
            already_calculated = full_history.encodings_in_memory(encodings)
            encodings = np.array(encodings)[~already_calculated]
            sequences_for_eval = np.array(sequences_for_eval)[~already_calculated]

            meta_data = {"debug" : self.debug, "step": int(step)}
            fitness, rewards, oracle_calls_exceeded = self.oracle_handler.get_fitness(encodings, meta_data = meta_data)

            if oracle_calls_exceeded:
                print("Oracle calls exceeded, stopping training")
                calls_in_current_batch = len(encodings)
                calls_before_this_batch = self.oracle_handler.oracle_calls - calls_in_current_batch
                slots_remaining = self.oracle_handler.max_oracle_calls - calls_before_this_batch
                slots_remaining = max(0, slots_remaining)

                encodings = encodings[:slots_remaining]
                fitness = fitness[:slots_remaining]
                sequences_for_eval = sequences_for_eval[:slots_remaining]

            meta_data = {"created_by": "agent_sampling", "step": int(step)}
            experience.add_experience_batch(sequences_for_eval, encodings, fitness, rewards, meta_data)
            #experience.write_memory(f"{experience_dir}/experience_memory_pure_{step}.csv")

            full_history.add_experience_batch(sequences_for_eval, encodings, fitness, rewards, meta_data)
            if self.debug:
                full_history.write_memory(f"{self.log_dir}/full_history.csv")
                plot_full_history(f"{self.log_dir}/full_history.csv", f"{self.log_dir}")
                if len(sequences_for_eval > 0):
                    analyze_final_history(f"{self.log_dir}/full_history.csv", 10, self.log_dir)
                else:
                    print(f"Warning: No new sequences were added to the experience memory this {step=}")

            if oracle_calls_exceeded:
                break

            #genetic_search
            if self.genetic_search and len(experience.memory) > self.population_size:
                print(f"Genetic search at step {step}")

                fitness = [experience.memory[i].fitness for i in range(len(experience.memory))]
                encodings = [experience.memory[i].encoding for i in range(len(experience.memory))]
                population = (encodings, fitness)

                for g in range(self.ga_generations):

                    child_encodings, child_crossover_stats, child_mutation_stats, pop_encodings, pop_scores = self.ga_handler.query(
                        query_size=self.offspring_size, mating_pool=population,
                        rank_coefficient=self.rank_coefficient,
                    )

                    if len(child_encodings) == 0:
                        print(f"Genetic search at step {step} created no children in generation {g}, skipping to next step")
                        continue

                    #filter for unique children
                    unique_child_encodings = unique(child_encodings)
                    child_encodings = child_encodings[unique_child_encodings]
                    child_crossover_stats = child_crossover_stats[unique_child_encodings]
                    child_mutation_stats = child_mutation_stats[unique_child_encodings]

                    # apply smarts filters
                    smarts_valid_mask = []
                    SMARTS_filtered_encodings_ga = []
                    drop_reasons_ga = []
                    for i, encoding in enumerate(child_encodings):
                        if self.encoding_type == "Smiles":
                            keep, reason = validate_molecule(encoding)
                        elif self.encoding_type == "GGS":
                            try:
                                mol_decoded = self.grammar.decoder(encoding)
                                smiles_decoded = Chem.MolToSmiles(mol_decoded)
                                keep, reason = validate_molecule(smiles_decoded)
                            except ValueError:
                                keep = False
                                reason = "Invalid SMILES"
                        if not self.use_SMARTS_filters:
                            keep = True
                        smarts_valid_mask.append(keep)
                        if keep:
                            SMARTS_filtered_encodings_ga.append(encoding)
                        else:
                            drop_reasons_ga.append(reason)

                    smarts_valid_mask = np.array(smarts_valid_mask, dtype=bool)

                    # Apply smarts mask to all relevant arrays
                    child_encodings = np.array(SMARTS_filtered_encodings_ga)
                    child_crossover_stats = child_crossover_stats[smarts_valid_mask]
                    child_mutation_stats = child_mutation_stats[smarts_valid_mask]

                    #check if child encodings have already been calculated
                    already_calculated = full_history.encodings_in_memory(child_encodings)
                    child_encodings = np.array(child_encodings)[~already_calculated]
                    child_crossover_stats = child_crossover_stats[~already_calculated]
                    child_mutation_stats = child_mutation_stats[~already_calculated]

                    if len(child_encodings) == 0:
                        print(f"All children removed by SMARTS filter or duplicate filter in generation {g}")
                        continue

                    #todo: genome object for group_selfies is created in ga_handler.query and oracle_handler.get_rewards --> inefficient
                    meta_data = {"debug": self.debug, "step": int(step), "generation": int(g)}
                    child_fitness, child_rewards, oracle_calls_exceeded = self.oracle_handler.get_fitness(child_encodings, meta_data = meta_data)
                    child_rewards["mutation_stats"] = child_mutation_stats
                    child_rewards["crossover_stats"] = child_crossover_stats
                    meta_data = {"created_by": "genetic_search", "step": int(step), "generation": int(g)}

                    assert len(child_encodings) == len(child_fitness), f"Number of child encodings {len(child_encodings)} does not match number of child fitness {len(child_fitness)}"

                    if oracle_calls_exceeded:
                        calls_in_current_batch = len(child_encodings)
                        calls_before_this_batch = self.oracle_handler.oracle_calls - calls_in_current_batch
                        slots_remaining = self.oracle_handler.max_oracle_calls - calls_before_this_batch
                        slots_remaining = max(0, slots_remaining)
                        child_encodings = child_encodings[:slots_remaining]
                        child_fitness = child_fitness[:slots_remaining]
                        #spaces_left = self.oracle_handler.oracle_calls - self.oracle_handler.max_oracle_calls - 2
                        #child_encodings = child_encodings[:spaces_left]
                        #child_fitness = child_fitness[:spaces_left]

                    experience.add_incomplete_experience_batch(child_encodings, child_fitness, child_rewards, meta_data)
                    #experience.write_memory(f"{experience_dir}/experience_memory_ga_search_{step}_generation_{g}_.csv")

                    full_history.add_incomplete_experience_batch(child_encodings, child_fitness, child_rewards, meta_data)
                    
                    if self.debug:
                        full_history.write_memory(f"{self.log_dir}/full_history.csv")
                        plot_full_history(f"{self.log_dir}/full_history.csv", f"{self.log_dir}")
                        analyze_final_history(f"{self.log_dir}/full_history.csv", 10, self.log_dir)

                    population = (pop_encodings + list(child_encodings), pop_scores + list(child_fitness))

            # Experience replay
            avg_loss = 0
            if self.experience_replay and len(experience.memory) > self.experience_replay:
                for replay_iteration in range(self.n_experience_iterations):
                    assert self.batch_size <= len(experience.memory), "Batch size is larger than memory size"

                    if not cooldown_active:
                        exp_sequences, experience_fitness, exp_smiles = experience.rank_based_sample(self.batch_size,
                                                                                     rank_coefficient=self.rank_coefficient,
                                                                                     return_memory_item=False,
                                                                                     return_smiles = True)
                    else:
                        #adapt the batch size during cooldown for experience replay
                        batch_size_cooldown = min(len(experience.memory), self.batch_size * 4)
                        exp_sequences, experience_fitness, exp_smiles = experience.uniform_sample(batch_size_cooldown,
                                                                                                     return_memory_item=False,
                                                                                                     return_smiles=True)

                    if torch.cuda.is_available():
                        fitness = torch.tensor(experience_fitness).cuda()
                        exp_sequences = torch.tensor(exp_sequences).cuda()
                    else:
                        fitness = torch.tensor(experience_fitness)
                        exp_sequences = torch.tensor(exp_sequences)

                    #check if exp_sequences is empty
                    if exp_sequences.shape[0] == 0:
                        print(f"Warning: No sequences sampled from experience memory at {step=}, {replay_iteration=}. Skipping...")
                        continue

                    padded_sequences, valid_mask = padding_and_valid_mask(exp_sequences, self.action_space, self.max_seq_length)
                    if self.use_max_seq_length_padding:
                        exp_sequences = padded_sequences

                    if self.use_masking == False:
                        valid_mask = torch.ones(exp_sequences.shape, dtype=torch.bool, device=exp_sequences.device)

                    exp_agent_likelihood, _, num_valid_steps_agent, descriptors = Agent.sequence_likelihood_for_pretraining(exp_sequences,
                                                                                               valid_mask=valid_mask,
                                                                                               use_masking=self.use_masking)
                    prior_agent_likelihood, _, num_valid_steps_prior, __ = Prior.sequence_likelihood_for_pretraining(exp_sequences,
                                                                                                 valid_mask=valid_mask,
                                                                                                 use_masking=self.use_masking)

                    exp_forward_flow = exp_agent_likelihood + log_z
                    exp_backward_flow = fitness * self.beta

                    print(f"Flow statistics: {exp_agent_likelihood=}, {log_z=}, {fitness * self.beta}")

                    loss = torch.pow(exp_forward_flow - exp_backward_flow, 2).mean()

                    # Kulback-Leibler divergence between Prior and Agent
                    loss_p = (exp_agent_likelihood - prior_agent_likelihood).mean()
                    loss += self.kl_coefficient * loss_p
                    print(f"Kl Loss (weighted) {self.kl_coefficient * loss_p}")

                    # Descriptor Loss
                    if desc_mean is not None and desc_std is not None and self.descriptor_weight > 0:
                        gt_list = []
                        valid_indices = []
                        for idx, smiles in enumerate(exp_smiles):
                            try:
                                if smiles:
                                    mol = Chem.MolFromSmiles(smiles)
                                    if mol:
                                        # Calculate descriptors using the new helper method
                                        vals = calc_descriptors(mol)
                                        if vals is not None:
                                            gt_list.append(vals)
                                            valid_indices.append(idx)
                            except Exception:
                                continue

                        if len(valid_indices) == len(exp_smiles):
                            gt_descriptors = torch.tensor(np.array(gt_list), device=device, dtype=torch.float)
                            gt_descriptors_normalized = (gt_descriptors - desc_mean) / desc_std
                            descriptor_loss = torch.nn.functional.mse_loss(descriptors, gt_descriptors_normalized)
                            loss += descriptor_loss * self.descriptor_weight
                            print(f"Loss {loss=}; descriptor_loss (weighted) {descriptor_loss * self.descriptor_weight}")
                        else:
                            raise ValueError("Descriptor calculation failed for some molecules in the batch. Ensure all SMILES are valid and descriptors can be computed.")


                    avg_loss += loss.item() / self.n_experience_iterations

                    optimizer.zero_grad()
                    loss.backward()

                    #logging of gradient norms
                    total_norm = clip_grad_norm_(Agent.net.parameters(), max_norm = self.gradient_clipping)
                    print(f"{step=} {replay_iteration=} {total_norm:.4f}")
                    if self.debug:
                        try:
                            tmp_path = os.path.join(self.log_dir, "gradient_norms.txt")
                            with open(tmp_path, 'a') as f:
                                f.write(f"{step=} {replay_iteration=} {total_norm.item():.4f}\n") # .item() gets the Python number
                        except IOError as e:
                            pass

                    optimizer.step()
            if self.debug:
                experience.write_memory(f"{experience_dir}/experience_memory.csv")

            #checkpointing
            if ckpt_handler.slurm_time_limit_approaching():
                ckpt_handler.save(step, Agent, optimizer, log_z, self.oracle_handler, experience, full_history, self.rng)
                print("Exiting due to time limit.")
                sys.exit(0)
            if self.oracle_handler.oracle_calls >= next_save_threshold:
                print(f"Oracle calls ({self.oracle_handler.oracle_calls}) reached threshold ({next_save_threshold}). Saving...")
                ckpt_handler.save(step, Agent, optimizer, log_z, self.oracle_handler, experience, full_history, self.rng)
                current_calls = self.oracle_handler.oracle_calls
                next_save_threshold = ((current_calls // save_interval) + 1) * save_interval
                print(f"Next checkpoint target: {next_save_threshold}")

            if oracle_calls_exceeded:
                print("Oracle calls exceeded, stopping training")
                break

        ckpt_handler.save(self.n_steps, Agent, optimizer, log_z, self.oracle_handler, experience, full_history, self.rng)
        full_history.write_memory(f"{self.log_dir}/full_history.csv")
        plot_full_history(f"{self.log_dir}/full_history.csv", f"{self.log_dir}")
        analyze_final_history(f"{self.log_dir}/full_history.csv", 10, self.log_dir)
        experience.write_memory(f"{experience_dir}/experience_memory_final.csv")
        torch.save(Agent.net.state_dict(), f"{self.log_dir}/agent_final.pt")







if __name__ == '__main__':
    config_file = args.config
    seed = args.seed
    trainer = GFlow_Mol_Trainer(config_file, seed)
    trainer.train()


