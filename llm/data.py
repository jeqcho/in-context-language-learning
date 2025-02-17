import numpy as np
import torch


def zipf(x: np.ndarray, alpha: float = 1) -> np.ndarray:
    """
    Compute the Zipf value for each element in x.

    Args:
        x (np.ndarray): Input array.
        alpha (float): Exponent parameter.

    Returns:
        np.ndarray: Zipf-transformed values.
    """
    return (x + 2.7) ** (-alpha)


def zipf_dist(n: int, alpha: float = 1) -> np.ndarray:
    """
    Generate a Zipf distribution over n elements.

    Args:
        n (int): Number of elements.
        alpha (float): Exponent parameter.

    Returns:
        np.ndarray: A probability distribution that sums to 1.
    """
    coeffs = zipf(np.arange(n), alpha)
    return coeffs / coeffs.sum()


def stationary_dist(prob: np.ndarray) -> np.ndarray:
    """
    Compute the stationary distribution for a Markov chain.

    Args:
        prob (np.ndarray): Square transition probability matrix.

    Returns:
        np.ndarray: The stationary distribution as a real-valued vector.
    """
    evals, evecs = np.linalg.eig(prob.T)
    evec1 = evecs[:, np.isclose(evals, 1)]
    evec1 = evec1[:, 0]
    stationary = evec1 / evec1.sum()
    return stationary.real


def batched_stationary_dist(probs: np.ndarray) -> np.ndarray:
    """
    Compute the stationary distributions for a batch of Markov chains.

    Args:
        probs (np.ndarray): Array of transition matrices with shape (B, N, N).

    Returns:
        np.ndarray: Array of stationary distributions with shape (B, N).
    """
    evals, evecs = np.linalg.eig(probs.transpose(0, 2, 1))
    evals, evecs = evals.real, evecs.real
    mask = np.isclose(evals, 1)
    evec1 = evecs * mask[:, np.newaxis, :]
    evec1 = evec1.sum(axis=2)
    stationary = evec1 / evec1.sum(axis=1, keepdims=True)
    return stationary.real


def batched_sinkhorn_knopp(
    matrices: np.ndarray, max_iter: int = 150, tol: float = 1e-9, cropped: bool = False
) -> np.ndarray:
    """
    Convert a batch of matrices to doubly stochastic matrices using the Sinkhorn-Knopp algorithm.

    Args:
        matrices (np.ndarray): Batch of matrices.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.
        cropped (bool): If True, replaces zero row/column sums with ones.

    Returns:
        np.ndarray: Batch of doubly stochastic matrices.
    """
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


def batched_multinomial(probs: np.ndarray) -> np.ndarray:
    """
    Sample from a batch of multinomial distributions.

    Args:
        probs (np.ndarray): Array of probabilities with shape (B, N) for each batch.

    Returns:
        np.ndarray: Array of sampled indices with shape (B,).
    """
    cumulative_probs = probs.cumsum(axis=1)
    random_values = np.random.rand(probs.shape[0], 1)
    samples = (random_values < cumulative_probs).argmax(axis=1)
    return samples


def get_random_state_tm(
    batch_size: int, min_states: int, max_states: int
) -> (np.ndarray, np.ndarray):
    """
    Generate a batch of random state transition matrices.

    Args:
        batch_size (int): Number of matrices to generate.
        min_states (int): Minimum number of states.
        max_states (int): Maximum number of states.

    Returns:
        tuple:
            - np.ndarray: Transition matrices of shape (batch_size, max_states, max_states).
            - np.ndarray: Number of states for each matrix.
    """
    num_states = np.random.randint(min_states, max_states + 1, size=batch_size)
    assert num_states.min() > 0, "Number of states must be greater than 0"
    transition_matrices = np.zeros((batch_size, max_states, max_states))
    for i in range(batch_size):
        n_states = num_states[i]
        dirichlet_samples = np.random.dirichlet(np.ones(n_states), size=n_states)
        transition_matrices[i, :n_states, :n_states] = dirichlet_samples
    return transition_matrices, num_states


def adjust_tm(
    P: np.ndarray,
    pi_target: np.ndarray,
    num_states: int,
    learning_rate: float = 0.01,
    max_iters: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Adjust a transition matrix so that its stationary distribution approximates the target distribution.

    Args:
        P (np.ndarray): Original transition matrix.
        pi_target (np.ndarray): Target stationary distribution.
        num_states (int): Number of states in P.
        learning_rate (float): Adjustment step size.
        max_iters (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.

    Returns:
        np.ndarray: Adjusted transition matrix.
    """
    P = P.copy()
    for iteration in range(max_iters):
        pi_current = stationary_dist(P)
        diff = pi_target - pi_current
        adjustment = learning_rate * diff

        for i in range(num_states):
            for j in range(num_states):
                if pi_current[i] > 0:
                    P[i, j] += adjustment[j] / pi_current[i]

        # Ensure non-negativity and renormalize rows
        P = np.maximum(P, 0)
        P /= P.sum(axis=1, keepdims=True)

        if np.linalg.norm(diff) < tol:
            break
    return P


def batch_adjust_tm(
    P_batch: np.ndarray,
    pi_target: np.ndarray,
    num_states: int,
    learning_rate: float = 0.01,
    max_iters: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Adjust a batch of transition matrices to approximate the target stationary distribution.

    Args:
        P_batch (np.ndarray): Batch of transition matrices with shape (B, N, N).
        pi_target (np.ndarray): Target stationary distribution.
        num_states (int): Number of states in each matrix.
        learning_rate (float): Adjustment step size.
        max_iters (int): Maximum iterations.
        tol (float): Tolerance for convergence.

    Returns:
        np.ndarray: Batch of adjusted transition matrices.
    """
    P_batch = P_batch.copy()
    adjusted = [
        adjust_tm(P_batch[i], pi_target, num_states, learning_rate, max_iters, tol)
        for i in range(P_batch.shape[0])
    ]
    return np.stack(adjusted)


def generate_batch(
    batch_size: int,
    lang_size: int,
    length: int,
    num_symbols: int,
    random_symbols: bool = False,
    low_symbols: int = None,
    high_symbols: int = None,
    random_selection: bool = False,
    low_idx: int = None,
    high_idx: int = None,
    doubly_stochastic: bool = False,
    zipfian: bool = False,
    eval: bool = False,
) -> torch.Tensor:
    """
    Generate a batch of sequences based on a Markov chain.

    Depending on the parameters, the function can generate transition matrices using random
    symbols or Dirichlet sampling, adjust them with Zipf or Sinkhorn-Knopp methods, and map states
    through a random selection process.

    Args:
        batch_size (int): Number of sequences to generate.
        lang_size (int): Number of different transition matrices.
        length (int): Length of the sequence.
        num_symbols (int): Number of symbols (states).
        random_symbols (bool): If True, generate random state transition matrices.
        low_symbols (int): Minimum states if random_symbols is True.
        high_symbols (int): Maximum states if random_symbols is True.
        random_selection (bool): If True, perform a random mapping of generated states.
        low_idx (int): Lower bound for token indices in random selection.
        high_idx (int): Upper bound for token indices in random selection.
        doubly_stochastic (bool): If True, adjust matrices to be doubly stochastic.
        zipfian (bool): If True, adjust matrices using a Zipf distribution.
        eval (bool): If True, return additional evaluation tensors.

    Returns:
        torch.Tensor:
            - If eval is False: a tensor of generated state tokens.
            - If eval is True: a tuple of tensors (states_tokens, states, tm, idxs).
    """
    # Generate transition matrices
    if random_symbols:
        if low_symbols is None or high_symbols is None:
            raise ValueError("low_symbols and high_symbols must be provided when random_symbols is True")
        tm, num_states_arr = get_random_state_tm(lang_size, low_symbols, high_symbols)
        num_symbols = high_symbols
    else:
        tm = np.random.dirichlet(np.ones(num_symbols), size=(lang_size, num_symbols))

    if zipfian:
        tm = batch_adjust_tm(tm, zipf_dist(num_symbols), num_symbols, learning_rate=0.01, max_iters=10000)

    if doubly_stochastic:
        tm = batched_sinkhorn_knopp(tm, cropped=random_symbols)

    # Extend transition matrices to the batch size by cycling through them
    cycle_idxs = np.arange(batch_size) % lang_size
    tm = tm[cycle_idxs]

    # Generate initial states from the stationary distribution
    sd = batched_stationary_dist(tm)
    states = [batched_multinomial(sd)]
    batch_indices = np.arange(batch_size)
    for _ in range(length):
        next_probs = tm[batch_indices, states[-1]]
        states.append(batched_multinomial(next_probs))
    states = np.stack(states, axis=1)

    # Optionally remap state indices
    if random_selection:
        if low_idx is None or high_idx is None:
            raise ValueError("low_idx and high_idx must be provided when random_selection is True")
        r = np.arange(low_idx, high_idx)
        idxs = np.array([np.random.choice(r, num_symbols, replace=False) for _ in range(batch_size)])
        dict_lst = [dict(zip(range(num_symbols), idxs[i])) for i in range(batch_size)]
        max_key = max(max(d.keys()) for d in dict_lst)
        value_maps = np.array([np.array([d.get(i, -1) for i in range(max_key + 1)]) for d in dict_lst])
        states_tokens = np.take_along_axis(value_maps[:, None, :], states[:, :, None], axis=2).squeeze(-1)
    else:
        states_tokens = states
        idxs = np.tile(np.arange(num_symbols), (batch_size, 1))

    if eval:
        return torch.tensor(states_tokens), torch.tensor(states), torch.tensor(tm), torch.tensor(idxs)
    else:
        return torch.tensor(states_tokens)


# Example usage:
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Generate a batch with evaluation outputs
    batch_tokens, batch_states, transition_matrices, idxs = generate_batch(
        batch_size=128,
        lang_size=16,
        length=100,
        num_symbols=35,
        random_symbols=False,
        low_symbols=None,
        high_symbols=None,
        random_selection=False,
        low_idx=None,
        high_idx=None,
        doubly_stochastic=False,
        zipfian=True,
        eval=True
    )
    print("Batch tokens shape:", batch_tokens.shape)