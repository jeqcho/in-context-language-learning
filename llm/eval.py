import torch

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def ground_truth(
    vocab_size: int,
    x_idx: torch.Tensor,
    tm,
    idx_to_token,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute the ground truth Markov probabilities.

    Args:
        vocab_size (int): Total vocabulary size.
        x_idx (torch.Tensor): Tensor of input indices with shape (B, T+1).
        tm: Transition matrix-like object that can be indexed as tm[b][x_idx[b], n].
        idx_to_token: Nested list mapping indices to token indices.
        device (torch.device): The torch device to use.

    Returns:
        torch.Tensor: A tensor of shape (B, T, vocab_size) containing the computed probabilities.
    """
    # Remove the last column of x_idx to align with tm indexing.
    x_idx = x_idx[:, :-1]

    B, T = x_idx.shape
    N = len(idx_to_token[0])
    markov_probs = torch.zeros(B, T, vocab_size, device=device)

    for b in range(B):
        for n in range(N):
            # For each sample and each token mapping, assign the probabilities.
            markov_probs[b, :, idx_to_token[b][n]] = tm[b][x_idx[b], n]
    return markov_probs


def unigram_strategy(
    seq_idx: torch.Tensor, num_states: int, device: torch.device
) -> torch.Tensor:
    """
    Compute unigram probability distribution from a sequence.

    Args:
        seq_idx (torch.Tensor): 1D tensor of state indices.
        num_states (int): Total number of states.
        device (torch.device): The torch device to use.

    Returns:
        torch.Tensor: A normalized probability distribution over states.
    """
    counts = torch.bincount(seq_idx, minlength=num_states).to(device)
    return counts / counts.sum()


def unigram_batch(
    batch: torch.Tensor, num_states: int, device: torch.device
) -> torch.Tensor:
    """
    Compute unigram distributions for each sequence in a batch.

    Args:
        batch (torch.Tensor): A batch of sequences (each a 1D tensor).
        num_states (int): Total number of states.
        device (torch.device): The torch device to use.

    Returns:
        torch.Tensor: A stacked tensor of unigram distributions.
    """
    distributions = [unigram_strategy(seq_idx, num_states, device) for seq_idx in batch]
    return torch.stack(distributions).to(device)


def bigram_strategy(
    seq_idx: torch.Tensor, num_states: int, device: torch.device
) -> torch.Tensor:
    """
    Compute a bigram probability distribution based on the last state.

    Args:
        seq_idx (torch.Tensor): 1D tensor of state indices.
        num_states (int): Total number of states.
        device (torch.device): The torch device to use.

    Returns:
        torch.Tensor: A normalized probability distribution over states based on bigram counts.
    """
    # Find indices where the state (except the last) equals the final state.
    last_state_matches = (
        torch.nonzero(seq_idx[:-1] == seq_idx[-1]).squeeze(-1).to(device)
    )
    if last_state_matches.numel() == 0:
        # If no match is found, return a uniform distribution.
        return torch.ones(num_states, device=device) / num_states
    next_state_indices = last_state_matches + 1
    next_states = seq_idx[next_state_indices]
    return unigram_strategy(next_states, num_states=num_states, device=device)


def bigram_batch(
    batch: torch.Tensor, num_states: int, device: torch.device
) -> torch.Tensor:
    """
    Compute bigram distributions for each sequence in a batch.

    Args:
        batch (torch.Tensor): A batch of sequences (each a 1D tensor).
        num_states (int): Total number of states.
        device (torch.device): The torch device to use.

    Returns:
        torch.Tensor: A stacked tensor of bigram distributions.
    """
    distributions = [bigram_strategy(seq_idx, num_states, device) for seq_idx in batch]
    return torch.stack(distributions).to(device)


def evaluator_logits(
    vocab_size: int,
    batched_probs: torch.Tensor,
    idx_to_token,
    device: torch.device,
) -> torch.Tensor:
    """
    Map batched probabilities to evaluator logits based on an index mapping.

    Args:
        vocab_size (int): Total vocabulary size.
        batched_probs (torch.Tensor): Tensor of shape (B, N) with probability values.
        idx_to_token: Nested list mapping for each batch sample.
        device (torch.device): The torch device to use.

    Returns:
        torch.Tensor: A tensor of shape (B, vocab_size) containing the logits.
    """
    B, N = batched_probs.shape
    logits = torch.zeros(B, vocab_size, device=device)

    for b, mapping in enumerate(idx_to_token):
        for n, token_idx in enumerate(mapping):
            logits[b, token_idx] = batched_probs[b, n]
    return logits
