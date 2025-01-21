# %%
from typing import Any, List
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Categorical
import numpy as np
from dataclasses import dataclass
import torch as t

#%%
device = t.device("cuda" if t.cuda.is_available() else "cpu")

# %%
@dataclass
class HMMArgs:
    num_emissions: int
    num_states: int
    seq_length: int


def init_model(hmm_args: HMMArgs) -> DenseHMM:
    hidden_states: List[Any] = [None] * hmm_args.num_states
    for i in range(hmm_args.num_states):
        hidden_states[i] = Categorical(
            t.full(size=(1, hmm_args.num_emissions), fill_value=1.0 / hmm_args.num_emissions)
        )
    edges = t.full(size=(hmm_args.num_states, hmm_args.num_states), fill_value=1.0 / hmm_args.num_states)
    starts = t.full(size=(hmm_args.num_states,), fill_value=1.0 / hmm_args.num_states)
    ends = t.full(size=(hmm_args.num_states,), fill_value=1.0 / hmm_args.num_states)

    model = DenseHMM(hidden_states, edges=edges, starts=starts, ends=ends, verbose=True)
    return model


# %%

if __name__ == "__main__":
    # init params
    hmm_args = HMMArgs(num_emissions=100, num_states=100, seq_length=100)

    # get training data
    train_fname = f"/n/netscratch/sham_lab/Everyone/jchooi/in-context-language-learning/data/TinyStories-{hmm_args.num_emissions}-train.txt"
    test_fname = f"/n/netscratch/sham_lab/Everyone/jchooi/in-context-language-learning/data/TinyStories-{hmm_args.num_emissions}-test.txt"

    with open(train_fname, "r") as f:
        train_lines = f.readlines()

    # concat into a big string and split into seq_length
    train_string = "".join(train_lines)
    train_string = train_string.replace("\n", "")
    train_integers = [int(token) for token in train_string.split(" ")]
    
    # log GPU
    print(f"Allocated memory before: {t.cuda.memory_allocated() / 1e9} GB")
    print(f"Reserved memory before: {t.cuda.memory_reserved() / 1e9} GB")

    # remove trailing sequence
    extra_length = len(train_integers) % hmm_args.seq_length
    train_integers = train_integers[:-extra_length]
    train_array = t.tensor(train_integers).reshape(-1, hmm_args.seq_length)

    # wrap each emission as 1d
    train_array = t.unsqueeze(train_array, -1)
    
    # train_array.shape is (65930, 100, 1)
    
    # demo
    train_array = train_array[:8000]
    
    # move tensor
    train_array = train_array.to(device)

    # init model
    model = init_model(hmm_args).to(device)
    
    # log GPU
    print(f"Allocated memory after init: {t.cuda.memory_allocated() / 1e9} GB")
    print(f"Reserved memory after init: {t.cuda.memory_reserved() / 1e9} GB")
    
    # train model
    model.fit(train_array)
    
    print(f"Allocated memory after train: {t.cuda.memory_allocated() / 1e9} GB")
    print(f"Reserved memory after train: {t.cuda.memory_reserved() / 1e9} GB")

    # save model
    model_fname = f"/n/netscratch/sham_lab/Everyone/jchooi/in-context-language-learning/models/hmm-H-{hmm_args.num_states}-E-{hmm_args.num_emissions}-L-{hmm_args.seq_length}.json"

    t.save(model, model_fname)
# %%
