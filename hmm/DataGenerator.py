"""
Utility class to use HMMs to generate synthetic data for LLMs
"""

from typing import List
import torch
from dataclasses import dataclass

from tqdm import tqdm
from hmm.utils import HMMWrapper
from numpy.typing import NDArray
import numpy as np
import random
import os


@dataclass
class DataGenerator:
    num_seq: int
    gen_seq_len: int
    permutate_emissions: bool
    hmm_wrapper: HMMWrapper
    epoch_on_filename: int
    suffix: str

    def __str__(self):
        perm_suffix = "-permutate_emissions" if self.permutate_emissions else ""
        suffix = f"-{self.suffix}" if self.suffix else ""
        return f"{str(self.hmm_wrapper)}-epoch_trained-{str(self.epoch_on_filename)}-gen_seq_len-{self.gen_seq_len}-num_seq-{self.num_seq:_}{perm_suffix}{suffix}"

    def __post_init__(self):
        self.data_filename = f"/n/netscratch/sham_lab/Everyone/jchooi/in-context-language-learning/data/synthetic/{self.__str__()}.npy"
        self.hmm_wrapper.model.sample_length = self.num_seq

    def generate_all(self, batch_size: int = 32) -> NDArray[np.int_]:
        # Use a temporary file for the memmap
        temp_filename = f"{self.data_filename}.tmp"

        try:
            # Create a memory-mapped array with the exact size we need
            final_shape = (self.num_seq, self.gen_seq_len)
            mmap_array = np.memmap(
                temp_filename, dtype=np.int32, mode="w+", shape=final_shape
            )

            # Generate batches and write to the memmap
            pbar = tqdm(total=self.num_seq, mininterval=0.5)
            sequences_generated = 0

            while sequences_generated < self.num_seq:
                if self.permutate_emissions:
                    # pomegranate uses ModuleList to store the emission probabilities
                    # we want to shuffle the distributions in each batch 
                    random.shuffle(self.hmm_wrapper.model.distributions)  # type: ignore

                # Calculate how many sequences to generate in this batch
                current_batch_size = min(batch_size, self.num_seq - sequences_generated)

                # Generate a batch
                current_batch = self.hmm_wrapper.model.batched_sample(
                    batch_size=current_batch_size, seq_len=self.gen_seq_len
                )

                # Write to memmap
                mmap_array[
                    sequences_generated : sequences_generated + current_batch_size
                ] = current_batch.cpu().numpy()

                # Update progress
                sequences_generated += current_batch_size
                pbar.update(current_batch_size)

            pbar.close()

            # Ensure all data is written
            mmap_array.flush()

            return mmap_array

        except Exception as e:
            print(f"Error during generation: {str(e)}")
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            raise

