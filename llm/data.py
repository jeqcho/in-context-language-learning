# %%
import numpy as np
import torch as t
from random import choices, sample
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
from tqdm import tqdm
from torch.utils.data import IterableDataset, DataLoader
# from utils import zipf_dist

def zipf(x, alpha=1):
    return (x + 2.7) ** (-alpha)


def zipf_dist(n, alpha=1):
    coeffs = zipf(np.arange(n), alpha)
    return coeffs / coeffs.sum()


def stationary_dist(prob):
  evals, evecs = np.linalg.eig(prob.T)
  evec1 = evecs[:,np.isclose(evals, 1)]
  evec1 = evec1[:,0]
  stationary = evec1 / evec1.sum()
  return stationary.real


def batched_stationary_dist(probs):
    evals, evecs = np.linalg.eig(probs.transpose(0, 2, 1))
    evals, evecs = evals.real, evecs.real
    evec1 = evecs * np.isclose(evals, 1)[:, np.newaxis, :]
    evec1 = evec1.sum(axis=2)
    stationary = evec1 / evec1.sum(axis=1)[:, np.newaxis]
    return stationary.real


def batched_sinkhorn_knopp(matrices, max_iter=150, tol=1e-9, cropped=False):
    A = np.copy(matrices)
    for _ in range(max_iter):
        row_sums = A.sum(axis=2, keepdims=True)
        if cropped:
            row_sums[row_sums == 0] = 1
        A /= row_sums
        col_sums = A.sum(axis=1, keepdims=True)
        if cropped:
            col_sums[col_sums == 0] = 1
        A /= col_sums
        if np.allclose(A.sum(axis=2), 1, atol=tol) and np.allclose(A.sum(axis=1), 1, atol=tol):
            break
    row_sums = A.sum(axis=2, keepdims=True)
    if cropped:
        row_sums[row_sums == 0] = 1
    A /= row_sums
    return A


def batched_multinomial(probs):
    cumulative_probs = probs.cumsum(axis=1)
    random_values = np.random.rand(probs.shape[0], 1)
    samples = (random_values < cumulative_probs).argmax(axis=1)
    return samples


def get_random_state_tm(batch_size, min_states, max_states):
    num_states = np.random.randint(min_states, max_states+1, size=batch_size)
    # range_states = np.arange(min_states, max_states+1)
    # dist = zipf_dist(max_states - min_states + 1, alpha=1/3)
    # num_states = np.random.choice(range_states, size=batch_size, p=dist)

    assert num_states.min() > 0
    transition_matrices = np.zeros((batch_size, max_states, max_states))
    for i in range(batch_size):
        n_states = num_states[i]
        dirichlet_samples = np.random.dirichlet(np.ones(n_states), size=n_states)
        transition_matrices[i, :n_states, :n_states] = dirichlet_samples
    return transition_matrices, num_states


def adjust_tm(P, pi_target, num_states, learning_rate=0.01, max_iters=1000, tol=1e-6):
    P = P.copy()
    for iteration in range(max_iters):
        pi_current = stationary_dist(P)
        diff = pi_target - pi_current

        # Compute the adjustment
        adjustment = learning_rate * diff

        # Apply the adjustment
        for i in range(num_states):
            for j in range(num_states):
                if pi_current[i] > 0:
                    P[i, j] += adjustment[j] / pi_current[i]
        
        # Normalize to ensure rows sum to 1
        P = np.maximum(P, 0)
        P /= P.sum(axis=1, keepdims=True)

        # Check for convergence
        if np.linalg.norm(diff) < tol:
            break
    return P


def batch_adjust_tm(P_batch, pi_target, num_states, learning_rate=0.01, max_iters=1000, tol=1e-6):
    P_batch = P_batch.copy()
    return np.stack([adjust_tm(P_batch[i], pi_target, num_states, learning_rate, max_iters, tol) for i in range(P_batch.shape[0])])


def generate_batch(batch_size, 
                   lang_size,
                   length, 
                   num_symbols, 
                   random_symbols=False, 
                   low_symbols=None, 
                   high_symbols=None, 
                   random_selection=False, 
                   low_idx=None, 
                   high_idx=None, 
                   doubly_stochastic=False, 
                   zipfian=False,
                   eval=False):
    if random_symbols:
        assert low_symbols is not None, 'low_symbols must be provided for random symbols'
        assert high_symbols is not None, 'high_symbols must be provided for random symbols'
        tm, num_states = get_random_state_tm(lang_size, low_symbols, high_symbols)
        num_symbols = high_symbols
    else:
        tm = np.random.dirichlet(np.ones(num_symbols), size=(lang_size, num_symbols, ))
    if zipfian:
        tm = batch_adjust_tm(tm, zipf_dist(num_symbols), num_symbols, learning_rate=0.01, max_iters=10000)
    if doubly_stochastic:
        tm = batched_sinkhorn_knopp(tm, cropped=random_symbols)

    # extend lang_size to batch_size by cycling the transition matrices
    cycle_idxs = np.arange(batch_size) % lang_size
    tm = tm[cycle_idxs]

    sd = batched_stationary_dist(tm)
    states = [batched_multinomial(sd)]
    range_tensor = np.arange(batch_size)
    for _ in range(length):
        next_probs = tm[range_tensor, states[-1]]
        states.append(batched_multinomial(next_probs))

    states = np.stack(states, axis=1)

    if random_selection:
        assert low_idx is not None, 'low_idx must be provided for random selection'
        assert high_idx is not None, 'high_idx must be provided for random selection'

        # sample without replacement num_symbols from low_idx to high_idx
        r = np.arange(low_idx, high_idx)
        idxs = np.array([np.random.choice(r, num_symbols, replace=False) for _ in range(batch_size)])
        dict_lst = [dict(zip(range(num_symbols), idxs[i])) for i in range(batch_size)]
        
        # efficiently replace the states with dict values
        max_key = max(max(d.keys()) for d in dict_lst)
        value_maps = np.array([np.array([d.get(i, -1) for i in range(max_key + 1)]) for d in dict_lst])
        states_tokens = np.take_along_axis(value_maps[:, None, :], states[:, :, None], axis=2).squeeze(-1)
    else:
        states_tokens = states
        idxs = np.tile(np.arange(0, num_symbols), batch_size).reshape(batch_size, num_symbols)

    if eval:
        return t.tensor(states_tokens), t.tensor(states), t.tensor(tm), t.tensor(idxs)
    else:
        return t.tensor(states_tokens)

# %%

# for i in tqdm(range(1000)):
#     x_t, x_idx, tm, idxs = generate_batch(batch_size=128, 
#                     lang_size=16,
#                     length=100, 
#                     num_symbols=35, 
#                     random_symbols=False, 
#                     low_symbols=None, 
#                     high_symbols=None, 
#                     random_selection=False, 
#                     low_idx=None, 
#                     high_idx=None, 
#                     doubly_stochastic=False, 
#                     zipfian=True,
#                     eval=True)

# %%
# batched_stationary_dist(tm)

# %%

# class MarkovDataset(IterableDataset):
#     def __init__(self, batch_size, num_symbols, length, random_selection=False, low_idx=None, high_idx=None, doubly_stochastic=False, eval=False):
#         self.batch_size = batch_size
#         self.num_symbols = num_symbols
#         self.length = length
#         self.random_selection = random_selection
#         self.low_idx = low_idx
#         self.high_idx = high_idx
#         self.doubly_stochastic = doubly_stochastic
#         self.eval=eval

#     def __iter__(self):
#         while True:
#             yield generate_batch(
#                 self.batch_size, 
#                 self.num_symbols, 
#                 self.length, 
#                 self.random_selection, 
#                 self.low_idx, 
#                 self.high_idx, 
#                 self.doubly_stochastic,
#                 self.eval
#             )

# %%
# generate_batch(1, 10, num_symbols=3, random_symbols=True, low_symbols=2, high_symbols=5, random_selection=True, low_idx=0, high_idx=30, eval=True)

# %%
# test_dl = DataLoader(MarkovDataset(128, 3, 100, random_selection=True, low_idx=0, high_idx=30, eval=True), batch_size=1, pin_memory=True)


# # %%
# test_batch = next(iter(test_dl))


# # %%
# for i in tqdm(range(1000)):
#     test_batch = next(iter(test_dl))

# # %%
# for i in tqdm(range(1000)):
#     test_batch = generate_batch(128, 3, 100, random_selection=True, low_idx=0, high_idx=30, eval=True)

# %%
# test_batch = generate_batch(128, 3, 100, random_selection=True, low_idx=0, high_idx=30)
# eval_batch = generate_batch(128, 3, 100, random_selection=True, low_idx=0, high_idx=30, eval=True)