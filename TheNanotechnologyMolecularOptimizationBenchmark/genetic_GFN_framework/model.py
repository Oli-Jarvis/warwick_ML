#inspired by https://github.com/hyeonahkimm/genetic_gfn/blob/main/sars_cov2/genetic_gfn/model.py
import torch
import torch.nn as nn
from torch.nn import init
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import padding_and_valid_mask
from action_space import Action_Space_GroupSelfies
from dataset import MolTensorDataset
import torch.nn.functional as F
from group_selfies import GroupGrammar
from transformer_utils import TransformerBlock, RMSNorm


def _init_weights(module):
    """
    Applies custom weight initialization to the model's modules.
    """
    if isinstance(module, nn.Linear):
        # Kaiming (He) initialization for Linear layers, suitable for models with ReLU/SiLU
        init.kaiming_normal_(module.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        # Normal distribution for Embedding layers
        init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.GRUCell):
        # Xavier for input-to-hidden weights
        init.xavier_uniform_(module.weight_ih)
        # Orthogonal for hidden-to-hidden weights
        init.orthogonal_(module.weight_hh)
        # Zeros for biases
        if module.bias_ih is not None:
            init.zeros_(module.bias_ih)
        if module.bias_hh is not None:
            init.zeros_(module.bias_hh)
    elif isinstance(module, RMSNorm):
        # Initialize the gain parameter to 1
        init.ones_(module.weight)


class ModernTransformer(nn.Module):
    def __init__(self, N_actions, d_model, n_head, n_kv_heads, num_layers, max_seq_len, max_batch_size, n_fingerprints):
        super().__init__()
        self.tok_embeddings = nn.Embedding(N_actions, d_model)
        ffn_hidden_dim = int(2 * (4 * d_model) / 3)
        self.max_seq_len = max_seq_len

        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_head, n_kv_heads, ffn_hidden_dim, max_seq_len, max_batch_size) for _ in range(num_layers)]
        )
        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, N_actions, bias=False)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # This can be a simple linear layer or a more complex MLP
        # self.fingerprint_head = nn.Linear(d_model, n_fingerprints)
        self.fingerprint_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, n_fingerprints)
        )

        # Pre-compute the causal mask for training/full context
        # The full mask is used when T > 1. Shape: (1, 1, max_len, max_len)
        mask = torch.full((1, 1, max_seq_len, max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, tokens: torch.Tensor, start_pos: int, use_causal_mask: bool = True, use_cache: bool = False) -> torch.Tensor:
        h = self.tok_embeddings(tokens)
        T = tokens.shape[1]

        causal_mask = None
        # Mask is only needed for the first full prompt run (if T > 1)
        if T > 1 and use_causal_mask:
            causal_mask = self.mask

        for layer in self.layers:
            # Pass start_pos to each layer
            h = layer(h, start_pos, causal_mask, use_cache)

        h = self.norm(h)
        logits = self.output(h)
        return logits

    def forward_for_pretraining(self, tokens: torch.Tensor):
        """
        Runs a forward pass for pre-training, adding a non-interfering
        [CLS] token to predict fingerprints.
        Apart from that, this is similar to the standard forward pass, with use_causal_mask=True, no caching and start_pos=0.
        """
        h_seq = self.tok_embeddings(tokens)  # [B, T, D]
        B, T, D = h_seq.shape

        # 1. Add CLS token
        # Expand the learnable token to match the batch size
        cls_vec = self.cls_token.expand(B, 1, -1)  # [B, 1, D]
        h_full = torch.cat([h_seq, cls_vec], dim=1)  # [B, T+1, D]

        # 2. Create the custom non-interfering mask
        # Shape: [1, 1, T+1, T+1]
        mask = torch.full((1, 1, T + 1, T + 1), float("-inf"), device=tokens.device, dtype=h_seq.dtype)

        # Causal mask for the sequence part [T, T]
        # This is the same as self.mask, but we build it dynamically for T
        causal_mask = torch.triu(torch.full((1, 1, T, T), float("-inf"), device=mask.device), diagonal=1)
        mask[:, :, :T, :T] = causal_mask

        # Let the CLS token (last row) see all sequence tokens
        mask[:, :, -1, :T] = 0.0
        # CLS token can also see itself
        mask[:, :, -1, -1] = 0.0
        # Note: The sequence tokens (rows :T) *cannot* see the CLS token (last column)

        # 3. Run through layers (no cache, start_pos=0)
        h = h_full
        for layer in self.layers:
            # We pass our custom mask here
            h = layer(h, start_pos=0, mask=mask, use_cache=False)

        # 4. Get outputs
        h_norm = self.norm(h)

        # Split the outputs for the sequence and the CLS token
        h_seq_out = h_norm[:, :-1, :]  # [B, T, D]
        h_cls_out = h_norm[:, -1, :]  # [B, D]

        # Get token logits and fingerprint predictions
        logits = self.output(h_seq_out)
        fingerprints = self.fingerprint_head(h_cls_out)

        return logits, fingerprints

    def clear_kv_cache(self):
        for layer in self.layers:
            layer.attention.cache_k.zero_()
            layer.attention.cache_v.zero_()


class MultiGRU(nn.Module):
    def __init__(self, num_layers, d_model, N_actions, n_fingerprints):
        super(MultiGRU, self).__init__()
        self.grus = nn.ModuleList()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(N_actions, 128)
        self.gru_1 = nn.GRUCell(128, d_model)
        for _ in range(num_layers - 1):
            self.grus.append(nn.GRUCell(d_model, d_model))
        self.linear = nn.Linear(d_model, N_actions)
        # self.fingerprint_head = nn.Linear(d_model, n_fingerprints)
        self.fingerprint_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, n_fingerprints)
        )
        print(f"Initialized with {num_layers=} and {len(self.grus)+1}")

    def forward(self, x, hidden_state):

        device = x.device

        x = self.embedding(x)
        hidden_state_out = torch.zeros(hidden_state.size(), device=device)

        x = hidden_state_out[0] = self.gru_1(x, hidden_state[0])
        for i, gru in enumerate(self.grus):
            x = hidden_state_out[i+1] = gru(x, hidden_state[i+1])  # First index is for gru_1

        x = self.linear(x)
        return x, hidden_state_out

    def forward_for_pretraining(self, x, hidden_state):
        device = x.device
        x = self.embedding(x)
        hidden_state_out = torch.zeros(hidden_state.size(), device=device)

        x = hidden_state_out[0] = self.gru_1(x, hidden_state[0])
        for i, gru in enumerate(self.grus):
            x = hidden_state_out[i+1] = gru(x, hidden_state[i+1])  # First index is for gru_1

        # 'x' is now the pre-logit, final hidden state from the last layer
        logits = self.linear(x)
        fingerprints = self.fingerprint_head(x)

        return logits, hidden_state_out, fingerprints

    def init_hidden_state(self, batch_size):
        # Initial cell state is zero
        temp = torch.zeros(self.num_layers, batch_size, self.d_model)
        return temp


class BaseModel():
    def __init__(self, action_space, batch_size, max_seq_length):
        print(f"{action_space.N_actions} actions available")
        self.action_space = action_space
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

        self.net = None  # nn.Module

    def to(self, device):
        self.net.to(device)
        return self

    def sequence_likelihood(self, sequence_batch, valid_mask, use_masking):
        raise NotImplementedError

    def sequence_likelihood_for_pretraining(self, sequence_batch, valid_mask, use_masking):
        raise NotImplementedError

    def sample_sequences(self, batch_size, temperature=1.0, use_masking = True, max_seq_length_padding = False):
        raise NotImplementedError


class ModernNLPTransformer(BaseModel):
    # Re-introduced n_kv_heads
    def __init__(self, action_space, batch_size, max_seq_length,
                 d_model=256, n_head=8, n_kv_heads=2, num_layers=4, n_fingerprints=5):
        super(ModernNLPTransformer, self).__init__(action_space, batch_size, max_seq_length)
        self.max_batch_size = 1024 # Pre-allocate KV cache for this max batch size
        assert batch_size <= self.max_batch_size, f"Batch size {batch_size} exceeds max KV cache batch size {self.max_batch_size}"
        self.net = ModernTransformer(
            N_actions=self.action_space.N_actions,
            d_model=d_model, n_head=n_head, n_kv_heads=n_kv_heads, num_layers=num_layers,
            max_seq_len=self.max_seq_length + 1,
            max_batch_size=self.max_batch_size,
            n_fingerprints=n_fingerprints,
        )
        print(f"Modern NLP Transformer initialized. Heads: {n_head} (Q), {n_kv_heads} (K/V). KV-Cache enabled.")

    def sequence_likelihood(self, sequence_batch, valid_mask, use_masking):
        """ Calculates likelihood for training. Uses full sequence forward pass (start_pos=0). """

        batch_size = sequence_batch.size(0)
        start_token = torch.zeros(batch_size, 1, dtype=torch.long, device=sequence_batch.device)
        start_token[:] = self.action_space.reversed_action_space['Start']
        x = torch.cat((start_token, sequence_batch[:, :-1]), 1)

        # For training, start_pos is always 0. The full sequence is passed, and causality is handled by the mask.
        self.net.clear_kv_cache()
        logits = self.net(x, start_pos=0, use_causal_mask=True, use_cache=False)

        log_prob = F.log_softmax(logits, dim=2)
        prob = F.softmax(logits, dim=2)
        target_log_probs = torch.gather(log_prob, 2, sequence_batch.unsqueeze(2)).squeeze(2)
        log_probs = torch.sum(target_log_probs * valid_mask, dim=1)
        entropy = -torch.sum(prob * log_prob * valid_mask.unsqueeze(2).float(), dim=[1, 2])
        num_valid_steps = torch.sum(valid_mask, 1).float()

        return log_probs, entropy, num_valid_steps

    def sequence_likelihood_for_pretraining(self, sequence_batch, valid_mask, use_masking):
        """
        Calculates likelihood, entropy, and fingerprints for pre-training.

        This method uses the 'forward_for_pretraining' call on the network
        to get both token logits and fingerprint predictions from the
        non-interfering [CLS] token.

        Args:
            sequence_batch : (batch_size, sequence_length) *Tensor of action sequences*
            valid_mask : (batch_size, sequence_length) *Mask indicating valid steps*
            use_masking : *Whether to use masking*

        Returns:
            log_probs : (batch_size) *Log probabilities of the sequences*
            entropy : (batch_size) *Entropy of the predicted distributions*
            num_valid_steps : (batch_size) *Number of valid steps per sequence*
            predicted_fingerprints: (batch_size, n_fingerprints) *Predicted fingerprints*
        """

        # 1. Prepare input tokens (same as original sequence_likelihood)
        batch_size = sequence_batch.size(0)
        start_token = torch.zeros(batch_size, 1, dtype=torch.long, device=sequence_batch.device)
        start_token[:] = self.action_space.reversed_action_space['Start']
        x = torch.cat((start_token, sequence_batch[:, :-1]), 1)

        # 2. Call the new forward method
        # This returns logits for the sequence and the predicted fingerprints
        self.net.clear_kv_cache()
        logits, predicted_fingerprints = self.net.forward_for_pretraining(x)

        # 3. Calculate log-probs and entropy (same as original sequence_likelihood)
        log_prob = F.log_softmax(logits, dim=2)
        prob = F.softmax(logits, dim=2)
        target_log_probs = torch.gather(log_prob, 2, sequence_batch.unsqueeze(2)).squeeze(2)
        log_probs = torch.sum(target_log_probs * valid_mask, dim=1)
        entropy = -torch.sum(prob * log_prob * valid_mask.unsqueeze(2).float(), dim=[1, 2])
        num_valid_steps = torch.sum(valid_mask, 1).float()

        return log_probs, entropy, num_valid_steps, predicted_fingerprints

    def sample_sequences(self, batch_size, temperature=1.0, use_masking=True, max_seq_length_padding=False):
        """
        Efficiently samples sequences using the KV Cache.
        """
        device = next(self.net.parameters()).device
        start_token_idx = self.action_space.reversed_action_space['Start']
        end_token_idx = self.action_space.reversed_action_space['End']

        # 1. Clear KV Cache before a new generation
        self.net.clear_kv_cache()

        # Start with the initial 'Start' token
        current_token = torch.full((batch_size, 1), start_token_idx, dtype=torch.long, device=device)

        sequences = []
        log_probs = torch.zeros(batch_size, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        entropy = torch.zeros(batch_size, device=device)
        step_mask = torch.ones(batch_size, device=device).bool()
        valid_mask = []

        # 2. Loop for generation
        for step in range(self.max_seq_length):
            # Pass ONLY the current token (B, 1) and its position (step)
            # This is the fix for the OOM error.
            logits = self.net(current_token, start_pos=step, use_causal_mask=False, use_cache=True)

            # Logits are now (B, 1, N_actions), so we squeeze the sequence dimension
            logits = logits.squeeze(1)

            prob = F.softmax(logits / temperature, dim=1)
            log_prob_dist = F.log_softmax(logits / temperature, dim=1)
            new_x = torch.multinomial(prob, num_samples=1).view(-1)

            end_token_sampled = (new_x == end_token_idx)

            if use_masking:
                step_mask = step_mask.bool() & (~end_token_sampled.bool())
                seq_to_append = new_x * step_mask + end_token_idx * (~step_mask.bool())
            else:
                seq_to_append = new_x

            sequences.append(seq_to_append.view(-1, 1))
            valid_mask.append(step_mask.view(-1, 1))

            log_probs += NLLLoss(log_prob_dist, new_x) * step_mask.float()  # Apply mask to accumulated log_probs
            entropy += -torch.sum((log_prob_dist * prob), 1) * step_mask.float()  # Apply mask to entropy

            # Update finished status
            finished = torch.ge(finished + end_token_sampled.data, 1)
            if torch.prod(finished) == 1:
                break

            # 3. Update 'current_token' to be the *newly sampled* token for the next loop
            current_token = seq_to_append.unsqueeze(1).detach()

            # Final assembly (unchanged)
        sequences = torch.cat(sequences, 1)
        valid_mask = torch.cat(valid_mask, 1).bool()

        if max_seq_length_padding == False:
            return sequences.data, valid_mask, log_probs, entropy

        padded_sequences, valid_mask = padding_and_valid_mask(sequences, self.action_space, self.max_seq_length)

        return padded_sequences, valid_mask, log_probs, entropy


class RNN(BaseModel):
    def __init__(self, num_layers, action_space, batch_size, max_seq_length, d_model, n_fingerprints):
        super(RNN, self).__init__(action_space, batch_size, max_seq_length)
        self.d_model = d_model
        self.net = MultiGRU(num_layers, d_model, action_space.N_actions, n_fingerprints)
        self.num_layers = num_layers
        self.n_fingerprints = n_fingerprints
        print("RNN initialized. Max seq length", self.max_seq_length)

    def sequence_likelihood(self, sequence_batch, valid_mask, use_masking):
        """
        Predict the likelihood of a sequence batch using the RNN.
        Args:
            sequence_batch : (batch_size, sequence_length) *Tensor of action sequences*
            valid_mask : (batch_size, sequence_length) *Mask indicating valid steps in the sequences (including one end token)*
            use_masking : *Whether to use masking for padded sequences*
        Returns:
            loss : *Negative log likelihood loss of the sequences*
            entropy : *Entropy of the predicted action distributions*
            num_valid_steps : *Number of valid steps in the sequences (for normalization)*

        """
        device = next(self.net.parameters()).device

        if use_masking == False:
            valid_mask = torch.ones(sequence_batch.shape, dtype=torch.bool, device=device)
        batch_size, sequence_length = sequence_batch.size()
        start_token = torch.zeros(batch_size, 1, dtype=torch.long, device=sequence_batch.device)
        start_token[:] = self.action_space.reversed_action_space['Start']
        x = torch.cat((start_token, sequence_batch[:, :-1]), 1)
        hidden_state = self.net.init_hidden_state(batch_size).to(device)

        log_probs = torch.zeros(batch_size, dtype=torch.float32, device=device)
        entropy = torch.zeros(batch_size, device=device)

        end_token_index = self.action_space.reversed_action_space['End']
        if not use_masking:
            end_token_index = -1

        for step in range(sequence_length):
            logits, hidden_state = self.net(x[:, step], hidden_state)
            log_prob = F.log_softmax(logits, dim = 1)
            prob = F.softmax(logits, dim = 1)

            #masking for proper handling of padded sequences
            step_mask = valid_mask[:, step]

            #log_probs += NLLLoss(log_prob, sequence_batch[:, step]) * step_mask
            if use_masking:
                log_probs[step_mask] += NLLLoss(log_prob[step_mask], sequence_batch[step_mask, step], end_token_index, end_token_weight = 1)
            else:
                log_probs[step_mask] += NLLLoss(log_prob[step_mask], sequence_batch[step_mask, step])

            entropy += -torch.sum(prob * log_prob, dim=1)

            #if all finished -> product over steps is 0
            all_finished = (step_mask == 0).all()
            if all_finished:
                break

        num_valid_steps = torch.sum(valid_mask, 1).float()

        return log_probs, entropy, num_valid_steps

    def sequence_likelihood_for_pretraining(self, sequence_batch, valid_mask, use_masking):
        """
        Calculates likelihood, entropy, and fingerprints for pre-training.

        This method uses the 'forward_for_pretraining' call on the network
        to get both token logits and fingerprint predictions from the
        last valid hidden state.
        """
        device = next(self.net.parameters()).device

        if use_masking == False:
            valid_mask = torch.ones(sequence_batch.shape, dtype=torch.bool, device=device)
        batch_size, sequence_length = sequence_batch.size()

        # Prepare inputs
        start_token = torch.zeros(batch_size, 1, dtype=torch.long, device=sequence_batch.device)
        start_token[:] = self.action_space.reversed_action_space['Start']
        x = torch.cat((start_token, sequence_batch[:, :-1]), 1)
        hidden_state = self.net.init_hidden_state(batch_size).to(device)

        # Initialize outputs
        log_probs = torch.zeros(batch_size, dtype=torch.float32, device=device)
        entropy = torch.zeros(batch_size, device=device)

        # Get the number of fingerprints from the head
        final_fingerprints = torch.zeros(batch_size, self.n_fingerprints, device=device)

        # Get the length of each sequence to find the last valid step
        seq_lengths = torch.sum(valid_mask, 1).long()

        end_token_index = self.action_space.reversed_action_space['End']
        if not use_masking:
            end_token_index = -1

        # Run the RNN step-by-step
        for step in range(sequence_length):
            # Call the new forward method
            logits, hidden_state, fingerprints = self.net.forward_for_pretraining(x[:, step], hidden_state)

            log_prob = F.log_softmax(logits, dim=1)
            prob = F.softmax(logits, dim=1)

            # Masking for proper handling of padded sequences
            step_mask = valid_mask[:, step]

            if use_masking:
                log_probs[step_mask] += NLLLoss(log_prob[step_mask], sequence_batch[step_mask, step], end_token_index, end_token_weight=1)
            else:
                log_probs[step_mask] += NLLLoss(log_prob[step_mask], sequence_batch[step_mask, step])

            entropy += -torch.sum(prob * log_prob, dim=1)

            # Store the fingerprint if this is the last valid step for an item
            is_last_step = (step == (seq_lengths - 1))
            if is_last_step.any():
                final_fingerprints[is_last_step] = fingerprints[is_last_step]

            # if all finished -> product over steps is 0
            all_finished = (step_mask == 0).all()
            if all_finished:
                break

        num_valid_steps = torch.sum(valid_mask, 1).float()

        return log_probs, entropy, num_valid_steps, final_fingerprints

    def sample_sequences(self, batch_size, temperature=1.0, use_masking = True, max_seq_length_padding = False):
        """
        Sample sequences from the RNN using a temperature parameter.
        Args:
            batch_size : *Batch size for sampling*
            temperature : *Temperature parameter for sampling*
            use_masking : *Whether to use masking for padded sequences*
            max_seq_length_padding : *Whether to pad sequences to max_seq_length. Note, that the sampling will always stop after all batch elements have sampled an end token*
        Returns:
            sequences : (batch_size, max_seq_length) *Sampled action sequences*
            valid_mask : (batch_size, max_seq_length) *Mask indicating valid steps in the sequences (including one end token)*
            log_probs : (batch_size) *Log probabilities of the sampled sequences*
            entropy : (batch_size) *Entropy of the predicted action distributions*
        """

        device = next(self.net.parameters()).device
        #start with start token -> needed?. Start token could be implicit
        start_token = torch.zeros(batch_size, dtype=torch.long, device=device)
        start_token[:] = self.action_space.reversed_action_space['Start']
        hidden_state = self.net.init_hidden_state(batch_size).to(device)
        x = start_token

        sequences = []
        log_probs = torch.zeros(batch_size, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        entropy = torch.zeros(batch_size, device=device)


        step_mask = torch.ones(batch_size, device = start_token.device).bool()
        valid_mask = []
        #mask which batch element is still sampling
        still_sampling = torch.ones(batch_size, device = start_token.device).bool()
        for step in range(self.max_seq_length):
            logits, hidden_state = self.net(x, hidden_state)
            prob = F.softmax(logits / temperature, dim = 1)
            log_prob = F.log_softmax(logits/temperature, dim=1)
            x = torch.multinomial(prob, num_samples=1).view(-1)

            #this part ensures that after an end token is sampled, only end tokens are sampled -> otherwise the model has to learn this itself
            end_token_index = self.action_space.reversed_action_space["End"]
            end_token_sampled = (x == end_token_index)
            if use_masking:
                x = torch.where(still_sampling, x, end_token_index)
                sequences.append(x.view(-1, 1))
                valid_mask.append(still_sampling.view(-1, 1))
                just_sampled_end = still_sampling & (x == end_token_index)
                still_sampling = still_sampling & ~just_sampled_end

            else:
                seq_to_append = x
                sequences.append(seq_to_append.view(-1, 1))
                valid_mask.append(step_mask.view(-1, 1))

            # the loss is the deviation from the actual action and the predicted action
            if use_masking:
                log_probs += NLLLoss(log_prob, x, end_token_index, 1)
            else:
                log_probs += NLLLoss(log_prob, x, -1, 1) * step_mask.float()
            entropy += -torch.sum((log_prob * prob), 1)

            #check if end tokes are sampled and all batch elements are finished
            #x = seq_to_append.detach()
            end_token_sampled = end_token_sampled.data
            finished = torch.ge(finished + end_token_sampled, 1)
            if torch.prod(finished) == 1:
                break

        sequences = torch.cat(sequences, 1)

        valid_mask = torch.cat(valid_mask, 1).bool()
        if max_seq_length_padding == False:
            return sequences.data, valid_mask, log_probs, entropy

        padded_sequences, valid_mask = padding_and_valid_mask(sequences, self.action_space, self.max_seq_length)

        #return sequences.data, valid_mask, log_probs, entropy
        return padded_sequences.data, valid_mask, log_probs, entropy


def NLLLoss(inputs, targets, end_token_index = -1, end_token_weight = 1):
    """
        Custom Negative Log Likelihood loss that returns loss per example,
        rather than for the entire batch. The loss is calculated based on a one-hot encoding of the target.

        Args:
            inputs : (batch_size, num_classes) *Log probabilities of each class*
            targets: (batch_size) *Target class index*
            end_token_index : *Index of the end token in the action space. If -1, no special handling for end token*
            end_token_weight : *Weight to apply to the end token loss. Default is 1 (no weighting)*

        Outputs:
            loss : (batch_size) *Loss for each example*
    """

    if torch.cuda.is_available():
        target_expanded = torch.zeros(inputs.size()).cuda()
    else:
        target_expanded = torch.zeros(inputs.size())

    #create one-hot encoding of the target
    target_expanded.scatter_(1, targets.contiguous().view(-1, 1).data, 1.0)
    loss = target_expanded * inputs #shape (batch_size, num_classes)
    if end_token_index != -1:
        #multiply loss with mask which is 5 for end token and 1 else
        loss = loss * (targets != end_token_index).unsqueeze(1).float() + loss * (targets == end_token_index).unsqueeze(1).float() * end_token_weight
    loss = torch.sum(loss, 1) #shape
    return loss


def get_model(model, action_space, batch_size, max_seq_length, num_layers=4, d_model=256, n_fingerprints=None):
    '''
    Factory function to get the desired model.
    Args:
        model (str): Type of model to create ('rnn' or 'transformer').
        action_space (Action_Space_GroupSelfies): The action space for the model.
        batch_size (int): Batch size for training/sampling.
        max_seq_length (int): Maximum sequence length for the model.
        num_layers (int): Number of layers in the model.
        d_model (int): Hidden dimension of the model.
    '''
    if n_fingerprints is None: n_fingerprints = 17
    if model == 'rnn':
        model = RNN(num_layers, action_space, batch_size, max_seq_length, d_model, n_fingerprints)
        num_params = sum(p.numel() for p in model.net.parameters() if p.requires_grad)
        print(f"RNN model with {num_layers} layers, {d_model} hidden dimensions and {num_params/1e6:.1f}M parameters initialized.")
    elif model == 'transformer':
        model = ModernNLPTransformer(action_space, batch_size, max_seq_length, num_layers=num_layers, d_model=d_model,
                                    n_head=8, n_kv_heads=2, n_fingerprints=n_fingerprints)
        num_params = sum(p.numel() for p in model.net.parameters() if p.requires_grad)
        print(f"Transformer model with {num_layers} layers, {d_model} hidden dimensions and {num_params/1e6:.1f}M parameters initialized.")
    else:
        raise ValueError(f"Unknown model type: {model}")

    # Apply the custom weight initialization
    #model.net.apply(_init_weights)
    #print("Weights randomly initialized")
    return model


if __name__ == '__main__':


    model_sizes = []  # [(name, param_names, num_params), ...]
    for model_name in ["rnn", "transformer"]:
        for d_model in [128, 256, 512, 1024]:
            for num_layer in [3, 4, 6, 8]:
                try:
                    grammar_path = "./data/GS_complex_grammar.txt"
                    grammar = GroupGrammar.from_file(grammar_path)
                    action_space = Action_Space_GroupSelfies(grammar)
                    model = get_model(model_name, action_space, batch_size=4, max_seq_length=100, num_layers=num_layer, d_model=d_model)
                    num_params = sum(p.numel() for p in model.net.parameters() if p.requires_grad)
                    model_sizes.append((f"{model_name}", f"{num_layer} layers, {d_model} d_model", num_params))
                except Exception as e:
                    print(f"Failed to initialize {model_name} with {num_layer} layers and {d_model} d_model: {e}")

    print("=== Model Sizes ===")
    for m in model_sizes:
        print(f"Model: {m[0]:<12} | Config: {m[1]:<20} | Params: {m[2]/1e6:.2f}M")
    print("==================================")



