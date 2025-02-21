"""
Uses HMMs to generate synthetic data for LLMs.
"""

# %%
from DataGenerator import DataGenerator
from utils import load_model, HMMWrapper, TimeTracker
from HMMArgs import HMMArgs

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

# track time
tracker = TimeTracker()

model = load_model(hmm_args, epoch_on_filename=20)
hmm_wrapper = HMMWrapper(model, hmm_args)

data_generator = DataGenerator(
    gen_seq_len=100,
    num_seq=int(1e4),
    permutate_emissions=False,
    hmm_wrapper=hmm_wrapper,
)

#%%
print("Generating data...")
result: NDArray[np.int_] = data_generator.generate_all(batch_size=64)
print("Data generation complete!")

print(f"Saving data to {data_generator.data_filename}")
np.save(data_generator.data_filename, result)
print("Data saved!")

# log time
print(f"Total time taken: {tracker.format_time(tracker.seconds_elapsed())}")
# %%
