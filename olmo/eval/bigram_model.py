"""A bigram model (designed for evaluating against other models with KL divergence)"""
import numpy as np

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

    
    def get_transition_matrix(self):
        return self.transition_matrix
