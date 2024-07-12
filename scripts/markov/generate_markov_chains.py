import torch as t
import numpy as np
from random import choices
from tqdm import tqdm

def stationary_distribution(prob):
    """
    Calculates the stationary distribution of a given transition probability matrix.

    Parameters:
    prob (numpy.ndarray): The transition probability matrix.

    Returns:
    numpy.ndarray: The stationary distribution vector.

    """
    evals, evecs = np.linalg.eig(prob.T)
    evec1 = evecs[:,np.isclose(evals, 1)]

    #Since np.isclose will return an array, we've indexed with an array
    #so we still have our 2nd axis.  Get rid of it, since it's only size 1.
    evec1 = evec1[:,0]

    stationary = evec1 / evec1.sum()

    #eigs finds complex eigenvalues and eigenvectors, so you'll want the real part.
    return stationary.real


def generate_markov_chain(num_symbols, seq_len, deterministic=False, doubly_stochastic=False):
    """
    Generate a sequence of symbols using a transition matrix.

    Args:
        num_symbols (int): The number of symbols in the sequence.
        length (int): The length of the sequence.
        deterministic (bool): Whether to use a deterministic transition matrix.
        doubly_stochastic (bool): Whether to use a doubly stochastic transition matrix.

    Returns:
        torch.Tensor: A tensor containing the generated sequence of symbols.
        np.ndarray: The transition matrix used for generating the sequence.
    """
    if deterministic:
        transition_matrix = np.zeros(shape=(num_symbols, num_symbols))
        perm = np.random.permutation(num_symbols)
        # use the permutation to generate the deterministic markov chain
        for i in range(num_symbols):
            transition_matrix[perm[i]][perm[(i+1)%num_symbols]] = 1
    else:
        transition_matrix = np.random.dirichlet(np.ones(num_symbols), size=(num_symbols,))
    
    if doubly_stochastic:
        # already so if deterministc
        if not deterministic:
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            col_sums = transition_matrix.sum(axis=0, keepdims=True)

            # Use the Sinkhorn-Knopp algorithm to make it doubly stochastic
            while not np.allclose(row_sums, 1, atol=1e-4) or not np.allclose(col_sums, 1, atol=1e-4):

                row_sums = transition_matrix.sum(axis=1, keepdims=True)
                col_sums = transition_matrix.sum(axis=0, keepdims=True)

                # Normalize rows
                transition_matrix /= row_sums
                # Normalize columns
                transition_matrix /=  col_sums

            # normalize rows one last time
            row_sum = transition_matrix.sum(axis=1, keepdims=True)
            transition_matrix /= row_sum

    stat_dist = t.tensor(stationary_distribution(transition_matrix))
    thresholds = transition_matrix.cumsum(axis = 1).tolist()
    inp = choices(range(num_symbols), weights = stat_dist)
    for i in range(seq_len-1):
        inp.extend(choices(range(num_symbols), cum_weights = thresholds[inp[-1]]))

    return t.tensor(inp, dtype = int), transition_matrix


def generate_markov_chains(num_seq, num_symbols, seq_len, deterministic=False):
    """
    Generate a batch of sequences using the data_gen function.

    Parameters:
    - num_seq (int): The number of sequences to generate in the batch.
    - seq_len (int): The length of each sequence.
    - num_symbols (int): The number of symbols in the sequence.
    - deterministic (bool): Whether to use a deterministic transition matrix.

    Returns:
    - torch.Tensor: A tensor containing the generated sequences, with shape (num_seq, seq_len).

    Example usage:
    >>> generate_markov_chains(2, 5, 3)
    tensor([[0, 1, 2, 1, 0],
        [2, 0, 1, 2, 1]])
    """
    return t.stack([generate_markov_chain(num_symbols, seq_len, deterministic)[0] for _ in range(num_seq)])


if __name__ == "__main__":
    # generate a json of 1 million markov chains each with seq len 150
    import json
    num_seq = 1000000
    seq_len = 150
    num_symbols = 3
    deterministic = False
    doubly_stochastic = True
    chains = []
    for i in tqdm(range(num_seq)):
        chain, _ = generate_markov_chain(num_symbols, seq_len, deterministic, doubly_stochastic)
        chains.append(''.join([str(x) for x in chain.tolist()]))
    with open("doubly_stochastic_markov_chains.json", "w") as f:
        json.dump(chains, f)
        