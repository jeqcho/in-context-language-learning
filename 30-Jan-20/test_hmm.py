"""
This script tests the HMM on the final token cross entropy.

Input
- /n/netscratch/sham_lab/Everyone/jchooi/in-context-language-learning/models/hmm-H-{hmm_args.num_states}-E-{hmm_args.num_emissions}-L-{hmm_args.seq_length}.pkl
"""

# %%
from utils import *

# %%
hmm_args_list = [
    HMMArgs(num_states=100, num_emissions=100, seq_length=100, num_epoch=10, batch_size=256),
    HMMArgs(num_states=200, num_emissions=100, seq_length=300, num_epoch=10, batch_size=256),
    HMMArgs(num_states=200, num_emissions=100, seq_length=300, num_epoch=20, batch_size=256),
    HMMArgs(num_states=200, num_emissions=100, seq_length=300, num_epoch=40, batch_size=256),
    HMMArgs(num_states=200, num_emissions=100, seq_length=600, num_epoch=10, batch_size=128),
    HMMArgs(num_states=200, num_emissions=100, seq_length=600, num_epoch=40, batch_size=128),
    HMMArgs(num_states=100, num_emissions=100, seq_length=100, num_epoch=2, batch_size=256),
    HMMArgs(num_states=300, num_emissions=100, seq_length=300, num_epoch=10, batch_size=128),
]

models_stats = dict()

#%%
for hmm_args in hmm_args_list:
    print(f"Testing with {hmm_args.num_states=}, {hmm_args.num_epoch=}, {hmm_args.seq_length=}, {hmm_args.num_epoch=}")
    model = torch.load(hmm_args.model_filename).to(device)
    hmm_wrapper = HMMWrapper(model, hmm_args)
    models_stats[hmm_args] = hmm_wrapper.get_final_token_statistics()

# %%
print(models_stats)

# %%
