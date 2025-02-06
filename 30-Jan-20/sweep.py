"""
For wandb sweeps to train HMM models and vary by number of hidden states, sequence length and epochs.

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

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
sweep_config = dict(
    method="grid",
    metric=dict(name="test-loss", goal="minimize"),
    parameters=dict(
        num_states=dict(values=[100, 200, 400]),
        seq_length=dict(values=[100, 200, 400]),
        num_epoch=dict(values=[10, 20, 40]),
    ),
)


def get_hmm_args_from_sweep(wandb_config) -> HMMArgs:
    num_states = wandb_config["num_states"]
    seq_length = wandb_config["seq_length"]
    num_epoch = wandb_config["num_epoch"]
    hmm_args = HMMArgs(
        num_emissions=100,
        num_states=num_states,
        seq_length=seq_length,
        batch_size=128,
        num_epoch=num_epoch,
    )
    return hmm_args


def train():
    # Define args & initialize wandb
    wandb.init(project="in-context-language-learning", reinit=False)

    # Train the model with these new hyperparameters (the second `wandb.init` call will be ignored)
    hmm_args = get_hmm_args_from_sweep(wandb.config)
    print(hmm_args)

    # init model
    model = init_model(hmm_args).to(device)
    hmm_wrapper = HMMWrapper(model, hmm_args)
    hmm_wrapper.train()


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_config, project="in-context-language-learning")
    wandb.agent(sweep_id=sweep_id, function=train, count=5)
    wandb.finish()

    # log GPU
    print(f"Reserved memory after init: {torch.cuda.memory_reserved() / 1e9} GB")

# %%
