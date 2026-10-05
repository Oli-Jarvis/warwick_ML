import configparser
import sys
import matplotlib.pyplot as plt
from rdkit import Chem, rdBase
#turn off rdkit warnings
rdBase.DisableLog('rdApp.error')
import os
import torch
import subprocess
from torch.utils.data import DataLoader
from tqdm import tqdm
from group_selfies import GroupGrammar
import random
import numpy as np
from utils import padding_and_valid_mask
from group_selfies import GroupGrammar


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
    parser.add_argument('--seed', type=int, help='seed', default = -1)
    parser.add_argument('config', type=str, help='Path to config file')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    seed = args.seed
    if seed != -1:
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        generator = torch.Generator()
        generator.manual_seed(seed)
        rng = np.random.default_rng(seed=seed)

from dataset import MolTensorDataset
from action_space import Action_Space_GroupSelfies, Action_Space_Smiles
from model import get_model
from GGS import GGS
from utils import Smiles_MolData


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def pretrain(config_file, seed = -1):
    """
    Pretrain the model using the MolTensorDataset.
    Args:
        config_file (string): Path to the configuration file.
    """

    config = configparser.ConfigParser()
    config.read(config_file)

    grammar_path = config.get("General", "grammar_path")
    grammar = GroupGrammar(grammar_path)
    encoding_type = config.get("General", "encoding_type")
    max_seq_length = config.getint("General", "max_seq_length")
    num_layers = config.getint("General", "num_layers")
    d_model = config.getint("General", "d_model")
    model = config.get("General", "model")

    include_descriptors = config.getboolean("Dataset", "include_descriptors", fallback=False)

    epochs = config.getint("Pretrain", "epochs")
    batch_size = config.getint("Pretrain", "batch_size")
    output_path = config.get("Pretrain", "output_path")
    os.makedirs(output_path, exist_ok=True)  # Creates the output directory if it doesn't exist yet
    plot_loss = config.getboolean("Pretrain", "plot_loss")
    save_logs = config.getboolean("Pretrain", "save_logs")
    learning_rate = config.getfloat("Pretrain", "learning_rate")
    dataset_path = config.get("Pretrain", "dataset_path")
    # balance valid and descriptor loss
    loss_weight = config.getfloat("Pretrain", "loss_weight", fallback=0.0)  # 0 -> only valid loss, 1 -> only descriptor loss
    if not include_descriptors and loss_weight != 0.0:
        print('Warning: loss_weight is set but include_descriptors is False. Setting loss_weight to 0.0')
        loss_weight = 0.0
    print(f'Weight Valid prediction: {1-loss_weight}, Descriptor prediction: {loss_weight}')

    use_masking = config.getboolean("Pretrain", "use_masking", fallback = True)
    use_max_seq_length_padding = config.getboolean("Pretrain", "use_max_seq_length_padding", fallback = True)

    # print all option read from config
    print("--------- Configuration: ----------")
    for section in config.sections():
        for key, value in config.items(section):
            print(f"{section}.{key} = {value}")
    print("-----------------------------------")

    rng = np.random.default_rng(seed) if seed != -1 else np.random.default_rng()


    if encoding_type == "GGS":
        grammar = GroupGrammar.from_file(grammar_path)
        action_space = Action_Space_GroupSelfies(grammar)

        data = torch.load(dataset_path,
                          map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                          weights_only=False)
        # Calculate descriptor mean and std if descriptors are included
        if include_descriptors:
            all_descriptors = data.tensors[2]
            desc_mean = all_descriptors.mean(dim=0).to(device)
            desc_std = all_descriptors.std(dim=0).to(device) + 1e-6
            print(f"Descriptor Mean: {desc_mean}")
            print(f"Descriptor Std Dev: {desc_std}")
            num_descriptors = all_descriptors.shape[1]
        else:
            num_descriptors = None  # dummy value
        if data and isinstance(data, MolTensorDataset):
            print("Loaded dataset")
            # collate function missing

            if seed == -1:
                data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
            else:
                data_loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, generator=generator)

            meta_data = data_loader.dataset.meta_data_dict
            #max_length = meta_data["max_length"]
            #grammar_path = meta_data["grammar_path"]
            encoding_type_data = meta_data["encoding_type"]
            if encoding_type_data != encoding_type:
                raise ValueError(
                    f"Encoding type in dataset ({encoding_type_data}) does not match the one in config ({encoding_type})")
        else:
            raise ValueError("Data is not a MolTensorDataset")

    elif encoding_type == "Smiles":
        if include_descriptors:
            raise NotImplementedError("Descriptors not implemented for Smiles encoding")
        action_space = Action_Space_Smiles(grammar_path)

        moldata = Smiles_MolData(dataset_path, action_space)

        #subset for debugging
        #subset_size = 1024*80
        #indices = list(range(subset_size))
        #from torch.utils.data import Subset
        #subset = Subset(moldata, indices)
        #moldata = subset
        num_descriptors = None
        fill_value = action_space.reversed_action_space['End']
        if seed == -1:
            data_loader = DataLoader(moldata,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     drop_last=True,
                                     collate_fn=lambda batch: Smiles_MolData.collate_fn(batch, fill_value=fill_value))
        else:
            data_loader = DataLoader(moldata,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     drop_last=True,
                                     collate_fn=lambda batch: Smiles_MolData.collate_fn(batch, fill_value=fill_value),
                                     generator=generator)
        print(f'build DataLoader with length {len(data_loader)}')

    model = get_model(model, action_space, batch_size, max_seq_length, num_layers=num_layers, d_model=d_model, n_fingerprints=num_descriptors)
    Prior = model
    Prior.to(device)
    print("Build Model")

    if include_descriptors and 'desc_mean' in locals():
        Prior.net.register_buffer('desc_mean', desc_mean)
        Prior.net.register_buffer('desc_std', desc_std)

    optimizer = torch.optim.Adam(Prior.net.parameters(), lr=learning_rate)

    # Loss function for descriptor prediction
    descriptor_loss_fn = torch.nn.MSELoss()

    valid_ratio = []
    n_atoms = []
    interpretable_loss_history = []
    print("Starting epochs")
    for epoch in range(1, epochs):
        pbar = tqdm(enumerate(data_loader), total = len(data_loader))
        for step, batch in pbar:

            #sample from DataLoader
            if encoding_type == "GGS":
                sequences = batch[0].long()
                valid_mask = batch[1].bool()
                if include_descriptors:
                    descriptors = batch[2].float().to(device)
            elif encoding_type == "Smiles":
                sequences = batch.long()
                padded_sequences, valid_mask = padding_and_valid_mask(sequences, action_space, max_seq_length)
                if use_max_seq_length_padding:
                    sequences = padded_sequences
            else:
                raise ValueError(f"Unknown encoding type: {encoding_type}")

            if use_masking == False:
                valid_mask = torch.ones(sequences.shape, dtype=torch.bool, device=sequences.device)

            #Calculate loss (for valid prediction)
            log_props, entropy, num_valid_steps, predicted_fingerprints = Prior.sequence_likelihood_for_pretraining(sequences, valid_mask, use_masking)
            #one could introduce weights here: loss = - ((log_props * num_valid_steps) / torch.mean(num_valid_steps)).mean()
            #loss = - ((log_props * num_valid_steps) / torch.mean(num_valid_steps)).mean()
            valid_loss = - log_props.mean()

            # Calculate los for descriptor prediction
            if include_descriptors:
                # Normalize descriptors
                descriptors_normalized = (descriptors - desc_mean) / desc_std
                descriptor_loss = descriptor_loss_fn(predicted_fingerprints, descriptors_normalized)
                # Print descriptor loss
                pbar.set_description(f"Valid Loss: {valid_loss.item():.2f} Descriptor Loss: {descriptor_loss.item():.2f}")
                # Combine losses
                loss = (1 - loss_weight) * valid_loss + loss_weight * descriptor_loss
            else:
                pbar.set_description(f"Valid Loss: {valid_loss.item():.2f}")
                loss = valid_loss

            #update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                with torch.no_grad():
                    #reduce learning rate
                    decrease_by = 0.03
                    for param in optimizer.param_groups:
                        param['lr'] *= (1 - decrease_by)

                    sampled_sequences, valid_mask, likelihood, _ = Prior.sample_sequences(batch_size, use_masking=use_masking, max_seq_length_padding = use_max_seq_length_padding)
                    if use_max_seq_length_padding:
                        assert sampled_sequences.shape == (batch_size, max_seq_length), f"Sampled sequences have wrong shape {sampled_sequences.shape}"

                    valid = 0
                    valid_sequences = []
                    valid_action_sequences = []
                    sampled_action_sequences = sampled_sequences.detach().cpu().numpy().tolist()
                    for i, seq in enumerate(sampled_action_sequences):

                        #valid only with end token
                        has_end = action_space.has_end_token(seq)

                        if encoding_type == "GGS":
                            group_selfies = action_space.action_sequence_to_encoding(seq)
                            if group_selfies == "" or not has_end:
                                continue
                            if group_selfies.count("[") == group_selfies.count("]"):
                                try:
                                    GGS.from_grammar_path(group_selfies, grammar_path=grammar_path, rng=rng)
                                    valid += 1
                                    valid_sequences.append(group_selfies)
                                    valid_action_sequences.append(seq)
                                except Exception as e:
                                    pass
                        elif encoding_type == "Smiles":
                            smiles = action_space.action_sequence_to_encoding(seq)
                            mol = Chem.MolFromSmiles(smiles)
                            if mol:
                                valid += 1
                                has_end = action_space.has_end_token(seq)
                                valid_sequences.append(smiles)
                                valid_action_sequences.append(seq)
                                n_atoms.append(mol.GetNumAtoms())
                        else:
                            raise ValueError(f"Unknown encoding type: {encoding_type}")
                    valid_ratio.append(valid/len(sampled_sequences))
                    if include_descriptors:
                        # Calculate an interpretable descriptor loss (denormalize)
                        descriptors_unnormalized = (descriptors_normalized * desc_std) + desc_mean
                        predicted_unnormalized = (predicted_fingerprints * desc_std) + desc_mean


                        # deviation of each component in percent of the mean deviation
                        interpretable_descriptor_loss = torch.mean(torch.abs(predicted_unnormalized - descriptors_unnormalized) / desc_mean, dim=0)
                        interpretable_descriptor_string = ", ".join([f"{x*100:.2f}\%" for x in interpretable_descriptor_loss.tolist()])
                        interpretable_loss_history.append(interpretable_descriptor_loss.detach().cpu().numpy())
                        tqdm.write(f"Valid sequences: {valid}/{len(sampled_sequences)}, Interpretable Descriptor Loss (mean % deviation): {interpretable_descriptor_string}")
                    else:
                        tqdm.write(f"Valid sequences: {valid}/{len(sampled_sequences)}")
                    if save_logs:
                        #write valid sequences to file
                        tmp_path = os.path.join(output_path, f'valid_sequences_epoch_{epoch}_{step}.txt')
                        with open(tmp_path, "a") as f:
                            for seq in valid_sequences:
                                f.write(f"{seq}\n")
                    if save_logs:
                        #write valid sequences to file
                        tmp_path = os.path.join(output_path, f'valid_action_sequences_epoch_{epoch}_{step}.txt')
                        with open(tmp_path, "a") as f:
                            for action_seq in valid_action_sequences:
                                f.write(f"{action_seq}\n")

    tmp_path = os.path.join(output_path, 'prior.pt')
    torch.save(Prior.net.state_dict(), tmp_path)
    if plot_loss:
        plt.plot(valid_ratio)
        plt.ylabel("Valid ratio")
        plt.xlabel("Epoch")
        plt.savefig(f"{output_path}/valid_ratio.pdf")
        #save valid_ratio to file
        with open(f"{output_path}/valid_ratio.txt", "w") as f:
            for ratio in valid_ratio:
                f.write(f"{ratio}\n")

    if plot_loss and include_descriptors and len(interpretable_loss_history) > 0:
        history_arr = np.array(interpretable_loss_history)

        plt.clf()
        plt.plot(history_arr)
        plt.ylabel("Mean % Deviation")
        plt.xlabel("Validation Step")

        plt.legend([f"Desc {i}" for i in range(history_arr.shape[1])])

        plt.title("Interpretable Descriptor Loss Evolution")
        plt.savefig(f"{output_path}/descriptor_loss_evolution.pdf")

        # Save raw data to file
        np.savetxt(f"{output_path}/descriptor_loss_history.txt", history_arr)
        print(f"Saved descriptor loss plot and history to {output_path}")

    #histogram of n_atoms in dataset
    if encoding_type == "Smiles":
        plt.clf()
        plt.hist(n_atoms, bins=range(1, max(n_atoms) + 2), align='left', rwidth=0.8)
        plt.xlabel('Number of Atoms')
        plt.ylabel('Frequency')
        tmp_path = os.path.join(output_path, 'n_atoms_histogram.pdf')
        plt.savefig(tmp_path)
        #save n_atoms to file
        tmp_path = os.path.join(output_path, 'n_atoms.txt')
        with open(tmp_path, "w") as f:
            for n in n_atoms:
                f.write(f"{n}\n")




if __name__ == '__main__':
    config_file = args.config
    pretrain(config_file, seed)
