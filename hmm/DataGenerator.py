"""
Utility class to use HMMs to generate synthetic data for LLMs
"""

from typing import List
import torch
from dataclasses import dataclass

from tqdm import tqdm
from utils import HMMWrapper
from numpy.typing import NDArray
import numpy as np
import random


@dataclass
class DataGenerator:
    num_seq: int
    gen_seq_len: int
    permutate_emissions: bool
    hmm_wrapper: HMMWrapper

    def __str__(self):
        return f"{str(self.hmm_wrapper)}-gen_seq_len-{self.gen_seq_len}-num_seq-{self.num_seq}"

    def __post_init__(self):
        self.data_filename = f"/n/netscratch/sham_lab/Everyone/jchooi/in-context-language-learning/data/synthetic/{self.__str__()}.npy"
        self.hmm_wrapper.model.sample_length = self.num_seq
        if self.permutate_emissions:
            # pomegranate uses ModuleList to store the emission probabilities
            random.shuffle(self.hmm_wrapper.model.distributions)  # type: ignore

        # sampling in pomegranate requires the model to be on the cpu
        self.hmm_wrapper.model.cpu()

    def generate_all(self, batch_size: int = 32) -> NDArray[np.int_]:
        all_emission_tensors = []
        pbar = tqdm(total=self.num_seq)
        while pbar.n < self.num_seq:
            current_emission_tensors = self.hmm_wrapper.model.batched_sample(
                batch_size=batch_size, seq_len=self.gen_seq_len
            )
            assert current_emission_tensors.shape == (batch_size, self.gen_seq_len)
            all_emission_tensors.append(current_emission_tensors)
            pbar.update(batch_size)

        all_emission_tensors = torch.concat(all_emission_tensors)
        assert all_emission_tensors.ndim == 2
        assert all_emission_tensors.shape[1] == self.gen_seq_len
        return all_emission_tensors.cpu().numpy()

    def generate_batch(self) -> NDArray[np.int_]:
        emission_tensors: List[torch.Tensor] = self.hmm_wrapper.model.sample(
            self.num_seq
        )

        # move to cpu and convert to numpy
        emissions: NDArray[np.int_] = np.array(
            [tensor.cpu().numpy() for tensor in emission_tensors]
        )
        return emissions
