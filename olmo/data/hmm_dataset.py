from torch.utils.data import Dataset
from olmo.config import HMMDatasetConfig
from olmo.data.hmm_generator import generate_hmm_sequence
import random
import pickle
import torch
import numpy as np


class HMMDataset(Dataset):
    def __init__(self, hmm_dataset_config: HMMDatasetConfig, epoch_size: int):
        self.num_symbols = hmm_dataset_config.num_symbols
        self.seq_len = hmm_dataset_config.seq_len
        self.num_hidden_states = hmm_dataset_config.num_hidden_states
        self.epoch_size = epoch_size
        self.zipfian_ratio = hmm_dataset_config.zipfian_ratio
        self.zipfian_scale = hmm_dataset_config.zipfian_scale
        self.permutate = hmm_dataset_config.permutate
        self.custom_hmm = hmm_dataset_config.custom_hmm
        if self.custom_hmm:
            with open(self.custom_hmm, "rb") as f:
                self.hmm = pickle.load(f)

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        if self.custom_hmm:
            observed_chain = torch.tensor(np.squeeze(self.hmm.sample(self.seq_len)[0]))
            return {"input_ids": observed_chain}
            
        zipfian = True if random.uniform(0,self.zipfian_ratio) else False
        
        observed_chain, emission_probs, hidden_sequence = generate_hmm_sequence(
            num_symbols=self.num_symbols,
            num_hidden_states=self.num_hidden_states,
            seq_len=self.seq_len,
            zipfian_flag=zipfian,
            zipfian_scale=self.zipfian_scale,
            permutate=self.permutate
        )
        return {
            "input_ids": observed_chain,
            "metadata": {
                "emission_probs": emission_probs,
            },
        }
