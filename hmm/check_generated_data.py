"""
Sanity check the generated data of a HMM
"""

"""
Uses HMMs to generate synthetic data for LLMs.
"""

# %%
from DataGenerator import DataGenerator
from utils import load_model, HMMWrapper
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

# load model
epoch_on_filename = 20
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
print(f"Loading data from {data_generator.data_filename}")
result = np.load(data_generator.data_filename)
print("Data loaded!")

# %%

# import here since this block can be called directly after generate_data.py
from utils import Tokenizer
import torch


tokenizer_filename = "/n/netscratch/sham_lab/Everyone/jchooi/in-context-language-learning/data/tokenizer.json"
tokenizer = Tokenizer(tokenizer_filename)
data = torch.tensor(result[:20])
print(tokenizer.detokenize_batch(data))
# %%
