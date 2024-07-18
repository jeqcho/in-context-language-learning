"""A bigram model (designed for evaluating against other models with KL divergence)"""
import numpy as np
import torch

class BigramModel:
    def __init__(self, dim):
        self.dim = dim
        # row_sum is the sum across columns for each row of freq_matrix
        # transition_matrix is freq_matrix / row_sum
        self.reset()
    
    def reset(self):
        self.row_sum = np.zeros(shape=(self.dim))
        # start with uniform distribution
        self.transition_matrix = np.full(shape=(self.dim, self.dim), fill_value=1/self.dim)
    
    def update(self, prev_token, current_token):
        # assume that the tokens are normalized to be in the range [0, dim)
        assert prev_token < self.dim, f"prev_token out of range. prev_token: {prev_token}, dim: {self.dim}"
        assert current_token < self.dim, f"current_token out of range. current_token: {current_token}, dim: {self.dim}"

        self.transition_matrix[prev_token] *= self.row_sum[prev_token] + self.dim
        self.transition_matrix[prev_token][current_token] += 1
        self.row_sum[prev_token] += 1
        self.transition_matrix[prev_token] /= self.row_sum[prev_token] + self.dim
    
    def load(self, instance):
        assert max(instance) < self.dim, f"max(instance) out of range. max(instance): {max(instance)}, dim: {self.dim}"

        # initialize as frequency matrix
        self.transition_matrix = np.full(shape=(self.dim, self.dim), fill_value=1, dtype='float64')
        for pre, nex in zip(instance[:-1], instance[1:]):
            self.transition_matrix[pre][nex] += 1
        # normalize so that entries in any given row sums up to 1
        self.transition_matrix /= np.sum(self.transition_matrix,axis=1).reshape(-1,1)

    
    def get_transition_matrix(self):
        return self.transition_matrix


class BatchedBigramModel:
    def __init__(self, dim):
        self.dim = dim
    
    def load(self, batch):
        mx = torch.max(batch)
        assert mx < self.dim, f"max(batch) out of range. max(batch): {mx}, dim: {self.dim}"

        # initialize as frequency matrix
        self.transition_matrix = torch.full((batch.shape[0], self.dim, self.dim), fill_value=1, dtype=torch.float64).to(batch.device)
        for pre, nex in zip(batch[:, :-1].T, batch[:, 1:].T):
            # pre (BATCH_SIZE, ) and nex (BATCH_SIZE, )
            self.transition_matrix[torch.arange(batch.shape[0]), pre, nex] += 1
        
        self.transition_matrix /= torch.sum(self.transition_matrix,axis=2).reshape(batch.shape[0], -1, 1)

        # return the probabilities for the final token
        return self.transition_matrix[torch.arange(batch.shape[0]), batch[:,-1]]

