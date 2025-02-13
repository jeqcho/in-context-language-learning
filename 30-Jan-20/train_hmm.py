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
import time
from time import gmtime, strftime

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%

if __name__ == "__main__":
    torch.manual_seed(91)
    # receive args from commmand
    parser = argparse.ArgumentParser(
        description="Train a Hidden Markov Model (HMM) using the pomegranate library."
    )
    parser.add_argument("--num_emissions", type=int, required=True, help="Number of emissions in the HMM")
    parser.add_argument("--num_states", type=int, required=True, help="Number of states in the HMM")
    parser.add_argument("--seq_length", type=int, required=True, help="Length of the sequences used for training")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training")
    parser.add_argument("--num_epoch", type=int, required=True, help="Number of epochs for training")
    parser.add_argument("--update_freq", type=int, help="Number of batches before calling from_summaries.")
    parser.add_argument("--unique", action="store_true", help="Train on unique sentences only")
    parser.add_argument(
        "--save_epoch_freq",
        type=int,
        help="Save the model at that epoch frequency. If save_epoch_freq=5, save model after 5 epochs of training.",
    )
    parser.add_argument("--no_save", action="store_true", help="Do not save the model")

    args = parser.parse_args()
    args.save = not args.no_save
    if args.update_freq is None:
        args.update_freq = "all"
    else:
        assert args.update_freq > 0

    if args.no_save is None:
        assert args.save_epoch_freq is not None
    else:
        assert args.save_epoch_freq is None

    # init params
    hmm_args = HMMArgs(
        num_emissions=args.num_emissions,
        num_states=args.num_states,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        num_epoch=args.num_epoch,
        unique=args.unique,
        update_freq=args.update_freq,
    )

    # init model
    start_time = time.time()
    model = init_model(hmm_args).to(device)
    hmm_wrapper = HMMWrapper(model, hmm_args)

    # log GPU
    print(f"Reserved memory after init: {torch.cuda.memory_reserved() / 1e9} GB")

    hmm_wrapper.train(save_flag=args.save, save_freq=args.save_epoch_freq)

    # post-training
    end_time = time.time()
    time_taken = int(float(end_time) - float(start_time))
    time_taken = strftime("%H:%M:%S", gmtime(time_taken))
    print(f"{time_taken=}")
# %%
