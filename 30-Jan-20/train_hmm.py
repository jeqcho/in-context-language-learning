"""
This script trains a Hidden Markov Model (HMM) using the pomegranate library.

Output File:
The trained HMM is saved at:
"/n/netscratch/sham_lab/Everyone/jchooi/in-context-language-learning/models/hmm-H-{hmm_args.num_states}-E-{hmm_args.num_emissions}-L-{hmm_args.seq_length}.pkl"

- H: Number of states in the HMM
- E: Number of emissions in the HMM
- L: Length of the sequences used for training

"""
# %%
from typing import Any, Iterable, List, Tuple
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Categorical
from utils import HMMArgs
import torch as t
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np

# %%
device = t.device("cuda" if t.cuda.is_available() else "cpu")
# %%


def init_model(hmm_args: HMMArgs) -> DenseHMM:
    hidden_states: List[Any] = [None] * hmm_args.num_states
    rng = np.random.default_rng(42)
    for i in range(hmm_args.num_states):
        dist = rng.uniform(low=0,high=1,size=(1, hmm_args.num_emissions))
        dist = dist / dist.sum()
        hidden_states[i] = Categorical(
            t.tensor(dist).tolist()
        )
    edges = t.full(size=(hmm_args.num_states, hmm_args.num_states), fill_value=1.0 / hmm_args.num_states).tolist()
    starts = t.full(size=(hmm_args.num_states,), fill_value=1.0 / hmm_args.num_states).tolist()
    ends = t.full(size=(hmm_args.num_states,), fill_value=1.0 / hmm_args.num_states).tolist()

    model = DenseHMM(hidden_states, edges=edges, starts=starts, ends=ends, verbose=False)
    return model

def get_train_loader(hmm_args: HMMArgs) -> Tuple[Iterable, int]:
    # get training data
    train_fname = f"/n/netscratch/sham_lab/Everyone/jchooi/in-context-language-learning/data/TinyStories-{hmm_args.num_emissions}-train.txt"
    test_fname = f"/n/netscratch/sham_lab/Everyone/jchooi/in-context-language-learning/data/TinyStories-{hmm_args.num_emissions}-test.txt"

    with open(train_fname, "r") as f:
        train_lines = f.readlines()

    # concat into a big string and split into seq_length
    train_lines = [line.strip() for line in train_lines]
    train_string = " ".join(train_lines)
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
    
    train_dataset = TensorDataset(train_array, t.empty_like(train_array)) # labels are dummy tensors
    return DataLoader(train_dataset, batch_size=hmm_args.batch_size, shuffle=True), len(train_dataset)
# %%

if __name__ == "__main__":
    # init params
    hmm_args = HMMArgs(num_emissions=100, num_states=400, seq_length=300, batch_size=128, num_epoch=10)
    save_flag = False
    print(hmm_args)

    # init model
    model = init_model(hmm_args).to(device)

    # log GPU
    print(f"Reserved memory after init: {t.cuda.memory_reserved() / 1e9} GB")

    # train model
    train_loader, total_len = get_train_loader(hmm_args)
    for i in range(hmm_args.num_epoch):
        pbar = tqdm(total=total_len, desc=f"Epoch {i+1}")
        print(f"{model.distributions[0].probs[0][0]=}")
        for batch, _ in train_loader:
            batch = batch.to(device)
            # model.fit(batch)
            model.summarize(batch)
            pbar.update(batch.shape[0])
        model.from_summaries()
        print(f"Allocated memory after this epoch: {t.cuda.memory_allocated() / 1e9} GB")
        print(f"Reserved memory after this epoch: {t.cuda.memory_reserved() / 1e9} GB")
        pbar.close()

    # save model
    if save_flag:
        # t.save(model, hmm_args.model_filename[:-4]+"fit.pkl")
        t.save(model, hmm_args.model_filename)
# %%
