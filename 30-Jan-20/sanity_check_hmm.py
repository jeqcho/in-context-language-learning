"""
This notebook loads a trained HMM and prompts it for text generation for manual sanity checks.

Input
- /n/netscratch/sham_lab/Everyone/jchooi/in-context-language-learning/models/hmm-H-{hmm_args.num_states}-E-{hmm_args.num_emissions}-L-{hmm_args.seq_length}.pkl
"""

# %%
from typing import Any, Iterable, List
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Categorical
from utils import *
from dataclasses import dataclass
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import json
from jaxtyping import Float

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hmm_args = HMMArgs(num_emissions=100, num_states=200, seq_length=300, batch_size=256, num_epoch=1)

# load model
print(f"Loading model from {hmm_args.model_filename}")

model = torch.load(hmm_args.model_filename).to(device)

# load sample sentences
sample_sentences_filename = f"sentences-{hmm_args.num_emissions}.txt"
with open(sample_sentences_filename, "r") as f:
    sample_sentences: List[str] = f.readlines()

# %%
# strip newlines
sample_sentences = [sentence.strip() for sentence in sample_sentences]

# concat into seq_len
text = " ".join(sample_sentences)

# break into a batch
str_tokens = text.split(" ")
idx = 0
seqs: List[str] = []
while idx + hmm_args.seq_length < len(str_tokens):
    seq = str_tokens[idx : idx + hmm_args.seq_length]
    seqs.append(" ".join(seq))
    idx += hmm_args.seq_length

# tokenize
tokenizer_filename = "/n/netscratch/sham_lab/Everyone/jchooi/in-context-language-learning/data/tokenizer.json"
tokenizer = Tokenizer(tokenizer_filename)
tokenized_sentences = tokenizer.tokenize_batch(seqs)

# reshape to set emission dimension = 1
tokenized_sentences = tokenized_sentences.unsqueeze(-1)
assert tokenized_sentences.ndim == 3

# %%
# pass it to the HMM
# the HMM will assign probabilities for each hidden state for each token in the seq_len
# then we multiply by the emission matrix to get the logits
# hidden_state_probs: Float[torch.Tensor, "batch seq_len n_hidden_states"] = model.predict_proba(tokenized_sentences)
# print(hidden_state_probs)
# all the same
# %%
emission_log_probs: Float[torch.Tensor, "batch seq_len n_hidden"] = model._emission_matrix(tokenized_sentences)

prediction_log_probs: Float[torch.Tensor, "batch seq_len n_hidden"] = model.predict_log_proba(tokenized_sentences)

# check shapes
print(f"{emission_log_probs.shape=}")
print(f"{prediction_log_probs.shape=}")

# check equality
print(f"{torch.allclose(emission_log_probs, prediction_log_probs) =}")

# visual check
print(f"{emission_log_probs=}")
print(f"{prediction_log_probs=}")

#%%
# get the argmax
predicted_emissions = emission_log_probs.argmax(-1).cpu()
predicted_emissions_str = tokenizer.detokenize_batch(predicted_emissions)
for idx, sentence in enumerate(predicted_emissions_str):
    compare_sentences(seqs[idx], sentence)

#%%
# get the predicted hidden states
predicted_states = prediction_log_probs.argmax(-1).cpu()
print(f"{predicted_states.shape=}")
assert predicted_states.shape == (len(seqs), hmm_args.seq_length)

# get the emission distribution for each predicted hidden state
def get_distributions_for_seq(seq):
    return [model.distributions[state].probs[0] for state in seq]

distributions = [get_distributions_for_seq(seq) for seq in predicted_states]

predicted_emission_distributions = torch.tensor(distributions, dtype=torch.float)
print(f"{predicted_emission_distributions.shape=}")
assert predicted_emission_distributions.shape == (len(seqs), hmm_args.seq_length, hmm_args.num_emissions)

#%%
predicted_emissions = predicted_emission_distributions.argmax(-1)

predicted_emissions_str = tokenizer.detokenize_batch(predicted_emissions)
for idx, sentence in enumerate(predicted_emissions_str):
    compare_sentences(seqs[idx], sentence)

# probs are same across emissions, but they are different across words. This shouldn't happen, since otherwise some of the probabilities don't sum to 1.

# %%
# sum the emissions to see if wrong
# emission_probs = torch.exp(emission_log_probs).squeeze(0).sum(-1)
# print(emission_probs)
# %%
