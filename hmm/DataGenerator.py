"""
Utility class to use HMMs to generate synthetic data for LLMs
"""

from typing import List
import torch
from dataclasses import dataclass
from hmm.utils import HMMWrapper
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

    def generate_batch(self) -> NDArray[np.int_]:
        emission_tensors: List[torch.Tensor] = self.hmm_wrapper.model.sample(
            self.num_seq
        )

        # move to cpu and convert to numpy
        emissions: NDArray[np.int_] = np.array(
            [tensor.cpu().numpy() for tensor in emission_tensors]
        )
        return emissions
