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
from utils import *
import torch
import argparse

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%

if __name__ == "__main__":
    # receive args from commmand
    parser = argparse.ArgumentParser(description="Train a Hidden Markov Model (HMM) using the pomegranate library.")
    parser.add_argument("--num_emissions", type=int, required=True, help="Number of emissions in the HMM")
    parser.add_argument("--num_states", type=int, required=True, help="Number of states in the HMM")
    parser.add_argument("--seq_length", type=int, required=True, help="Length of the sequences used for training")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training")
    parser.add_argument("--num_epoch", type=int, required=True, help="Number of epochs for training")
    parser.add_argument("--unique", action="store_true", help="Train on unique sentences only")
    
    args = parser.parse_args()
    # init params
    hmm_args = HMMArgs(num_emissions=args.num_emissions, num_states=args.num_states, seq_length=args.seq_length, batch_size=args.batch_size, num_epoch=args.num_epoch, unique=args.unique)

    # init model
    model = init_model(hmm_args).to(device)
    hmm_wrapper = HMMWrapper(model, hmm_args)

    # log GPU
    print(f"Reserved memory after init: {torch.cuda.memory_reserved() / 1e9} GB")

    hmm_wrapper.train()
# %%
