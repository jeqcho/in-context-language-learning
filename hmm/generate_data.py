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
    num_emissions=200,
    num_states=200,
    seq_length=100,
    batch_size=1024,
    num_epoch=1000,
    unique=False,
    update_freq=64,
)

# track time
tracker = TimeTracker()

# load model
epoch_on_filename = 10
model = load_model(hmm_args, epoch_on_filename=epoch_on_filename)
hmm_wrapper = HMMWrapper(model, hmm_args)

data_generator = DataGenerator(
    gen_seq_len=100,
    num_seq=int(1e5),
    permutate_emissions=False,
    hmm_wrapper=hmm_wrapper,
    epoch_on_filename=epoch_on_filename,
)

# %%
print("Generating data...")
result: NDArray[np.int_] = data_generator.generate_all(batch_size=64)
print("Data generation complete!")

print(f"Saving data to {data_generator.data_filename}")
np.save(data_generator.data_filename, result)
print("Data saved!")

# log time
print(f"Total time taken: {tracker.format_time(tracker.seconds_elapsed())}")
# %%
