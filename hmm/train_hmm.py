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
import torch
import time
from time import gmtime, strftime

from HMMArgs import HMMArgs
from utils import HMMWrapper, get_hmm_args_parser, init_model, load_model

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
if __name__ == "__main__":
    torch.manual_seed(91)
    # receive args from commmand
    parser = get_hmm_args_parser()
    parser.add_argument("--no_save", action="store_true", help="Do not save the model")
    parser.add_argument(
        "--load_model_with_epoch", type=int, help="Load the model from file."
    )

    args = parser.parse_args()
    args.save = not args.no_save

    if args.no_save:
        assert args.save_epoch_freq is None
    else:
        assert args.save_epoch_freq is not None

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
    if args.load_model_with_epoch is None:
        model = init_model(hmm_args).to(device)
    else:
        model = load_model(hmm_args, args.load_model_with_epoch)
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
