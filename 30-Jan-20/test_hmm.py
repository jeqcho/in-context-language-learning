"""
This script tests the HMM on the final token cross entropy.

Input
- /n/netscratch/sham_lab/Everyone/jchooi/in-context-language-learning/models/hmm-H-{hmm_args.num_states}-E-{hmm_args.num_emissions}-L-{hmm_args.seq_length}.pkl
"""

# %%
from typing import Iterable, List, Tuple

from tqdm import tqdm
from utils import *
import torch
from jaxtyping import Float
from torch.utils.data import TensorDataset, DataLoader


def get_test_loader(hmm_args: HMMArgs) -> Tuple[Iterable, int]:
    test_fname = f"/n/netscratch/sham_lab/Everyone/jchooi/in-context-language-learning/data/TinyStories-{hmm_args.num_emissions}-test.txt"

    with open(test_fname, "r") as f:
        test_lines = f.readlines()

    # concat into a big string and split into seq_length
    test_lines = [line.strip() for line in test_lines]
    test_string = " ".join(test_lines)
    test_integers = [int(token) for token in test_string.split(" ")]

    # log GPU
    print(f"Allocated memory before: {torch.cuda.memory_allocated() / 1e9} GB")
    print(f"Reserved memory before: {torch.cuda.memory_reserved() / 1e9} GB")

    # remove trailing sequence
    extra_length = len(test_integers) % hmm_args.seq_length
    train_integers = test_integers[:-extra_length]
    train_array = torch.tensor(train_integers).reshape(-1, hmm_args.seq_length)

    # wrap each emission as 1d
    train_array = torch.unsqueeze(train_array, -1)

    train_dataset = TensorDataset(train_array, torch.empty_like(train_array))  # labels are dummy tensors
    return DataLoader(train_dataset, batch_size=hmm_args.batch_size, shuffle=True), len(train_dataset)


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hmm_args = HMMArgs(num_emissions=100, num_states=100, seq_length=100, batch_size=256, num_epoch=10)

# load modelx
print(f"Loading model from {hmm_args.model_filename}")

model = torch.load(hmm_args.model_filename).to(device)
hmm_wrapper = HMMWrapper(model)

#%%
test_loader = get_test_loader(hmm_args)

#%%
# train model
test_loader, total_len = get_test_loader(hmm_args)
pbar = tqdm(total=total_len)
ce_list = []
for batch, _ in test_loader:
    batch = batch.to(device)
    ce = hmm_wrapper.get_final_token_cross_entropy(batch)
    ce_list.append(ce)
    pbar.update(batch.shape[0])
pbar.close()
# %%
ce_list = torch.tensor(ce_list)
print(f"{ce_list.mean()=}")
print(f"{ce_list.std()=}")

# %%
