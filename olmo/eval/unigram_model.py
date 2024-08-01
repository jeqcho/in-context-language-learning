"""A unigram model (designed for evaluating against other models with KL divergence)"""

import numpy as np
import torch
import logging

log = logging.getLogger(__name__)


class UnigramModel:
    def __init__(self, dim):
        self.dim = dim
        # row_sum is the sum across columns for each row of freq_matrix
        # transition_matrix is freq_matrix / row_sum
        self.reset()

    def reset(self):
        self.tokens_saw = 0
        # prob_table is a 1D list of probabilities for each token
        self.prob_table = np.full(shape=(self.dim), fill_value=1 / self.dim)

    def load(self, sequence):
        assert (
            max(sequence) < self.dim
        ), f"current_token out of range. current_token: {max(sequence)}, dim: {self.dim}"

        for token in sequence:
            self.prob_table[token] += 1

        self.tokens_saw = len(sequence)
        self.prob_table /= self.tokens_saw

    def update(self, current_token):
        # assume that the tokens are normalized to be in the range [0, dim)
        assert (
            current_token < self.dim
        ), f"current_token out of range. current_token: {current_token}, dim: {self.dim}"

        self.prob_table *= self.tokens_saw
        self.prob_table[current_token] += 1
        self.tokens_saw += 1
        self.prob_table /= self.tokens_saw

    def get_probability_table(self):
        return self.prob_table


class BatchedUnigramModel:
    def __init__(self, dim):
        self.dim = dim

    def load(self, batch):
        mx = np.max(batch)
        assert mx < self.dim, f"max(batch) out of range. max(batch): {mx}, dim: {self.dim}"

        # initialize as frequency matrix
        self.prob_table = np.full((batch.shape[0], self.dim), fill_value=1, dtype=np.float64)
        for i in range(self.dim):
            self.prob_table[:, i] += np.sum(batch == i, axis=1)

        # equivalent function, maybe faster, test it out
        # for token in batch.T:
        #     self.prob_table[np.arange(batch.shape[0]), token] += 1

        self.prob_table /= np.sum(self.prob_table, axis=1).reshape(batch.shape[0], -1)

        return self.prob_table
