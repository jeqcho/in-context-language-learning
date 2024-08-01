from torch.utils.data import Dataset
from olmo.config import HMMDatasetConfig
from olmo.data.hmm_generator import generate_hmm_sequence
import numpy as np


class HMMDataset(Dataset):
    def __init__(self, hmm_dataset_config: HMMDatasetConfig, epoch_size: int):
        self.num_symbols = hmm_dataset_config.num_symbols
        self.seq_len = hmm_dataset_config.seq_len
        self.num_hidden_states = hmm_dataset_config.num_hidden_states
        self.epoch_size = epoch_size
        self.zipfian = hmm_dataset_config.zipfian
        self.zipfian_scale = hmm_dataset_config.zipfian_scale

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        observed_chain, emission_probs, hidden_sequence = generate_hmm_sequence(
            num_symbols=self.num_symbols,
            num_hidden_states=self.num_hidden_states,
            seq_len=self.seq_len,
            zipfian_flag=self.zipfian,
            zipfian_scale=self.zipfian_scale,
        )
        return {
            "input_ids": observed_chain,
            "metadata": {
                "emission_probs": emission_probs,
            },
        }
