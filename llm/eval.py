# %%
import torch as t

device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# %%
def ground_truth(vocab_size, x_idx, tm, idx_to_token, device):
    x_idx = x_idx[:, :-1]

    C = vocab_size
    B, T = x_idx.shape
    N = len(idx_to_token[0])

    markov_probs = t.zeros(B, T, C).to(device)
    for b in range(B):
        for n in range(N):
            markov_probs[b, :, idx_to_token[b][n]] = tm[b][x_idx[b], n]
    return markov_probs


def unigram_strategy(seq_idx: t.tensor, num_states, device):
    counts = t.bincount(seq_idx, minlength=num_states).to(device)
    return counts / counts.sum()


def unigram_batch(batch: t.tensor, num_states, device):
    return t.stack(
        [unigram_strategy(seq_idx, num_states, device) for seq_idx in batch]
    ).to(device)


def bigram_strategy(seq_idx: t.tensor, num_states, device):
    last_state_idx = t.nonzero(seq_idx[:-1] == seq_idx[-1]).squeeze(-1).to(device)
    if last_state_idx.numel() == 0:
        return (t.ones(num_states) / num_states).to(device)
    next_state_idx = last_state_idx + 1
    next_states = seq_idx[next_state_idx]
    return unigram_strategy(next_states, num_states=num_states, device=device)


def bigram_batch(batch: t.tensor, num_states, device):
    return t.stack(
        [bigram_strategy(seq_idx, num_states, device) for seq_idx in batch]
    ).to(device)


def evaluator_logits(vocab_size, batched_probs, idx_to_token, device):
    B, N = batched_probs.shape
    C = vocab_size
    markov_probs = t.zeros(B, C).to(device)

    for b in range(B):
        for n in range(N):
            markov_probs[b, idx_to_token[b][n]] = batched_probs[b, n]
    return markov_probs
