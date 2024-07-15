import torch
from torch.utils.data import Dataset
from olmo.config import MarkovDatasetConfig
from olmo.data.markov_chain_generator import generate_markov_chain


class MarkovDataset(Dataset):
    def __init__(self, markov_dataset_config: MarkovDatasetConfig, epoch_size: int):
        self.num_states = markov_dataset_config.num_states
        self.seq_len = markov_dataset_config.seq_len
        self.vocab_size = markov_dataset_config.vocab_size
        self.epoch_size = epoch_size

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        chain, tmatrix, chosen_symbols = generate_markov_chain(
            num_symbols=self.num_states, seq_len=self.seq_len, vocab_size=self.vocab_size
        )
        return {"input_ids": chain, "metadata": {"chosen_symbols": chosen_symbols}}  # for n-gram evaluations
