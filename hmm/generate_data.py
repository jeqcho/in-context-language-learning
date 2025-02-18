"""
Uses HMMs to generate synthetic data for LLMs.
"""

# %%
from hmm.DataGenerator import DataGenerator
from hmm.utils import load_model, HMMWrapper
from hmm.HMMArgs import HMMArgs

from numpy.typing import NDArray
import numpy as np

hmm_args = HMMArgs(
    num_emissions=100,
    num_states=500,
    seq_length=100,
    batch_size=256,
    num_epoch=1000,
    unique=False,
    update_freq=32,
)


model = load_model(hmm_args, epoch_on_filename=5)
hmm_wrapper = HMMWrapper(model, hmm_args)

data_generator = DataGenerator(
    gen_seq_len=100,
    num_seq=int(1e9),
    permutate_emissions=False,
    hmm_wrapper=hmm_wrapper,
)

print("Generating data...")
result: NDArray[np.int_] = data_generator.generate_batch()
print("Data generation complete!")

print(f"Saving data to {data_generator.data_filename}...")
np.save(data_generator.data_filename, result)
print("Data saved!")
# %%
