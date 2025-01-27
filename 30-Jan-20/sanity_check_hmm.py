"""
This notebook loads a trained HMM and prompts it for text generation for manual sanity checks.

Input
- /n/netscratch/sham_lab/Everyone/jchooi/in-context-language-learning/models/hmm-H-{hmm_args.num_states}-E-{hmm_args.num_emissions}-L-{hmm_args.seq_length}.pkl
"""

#%%
from typing import Any, Iterable, List
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Categorical
from utils import HMMArgs, Tokenizer
from dataclasses import dataclass
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import json
from jaxtyping import Float

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hmm_args = HMMArgs(num_emissions=200, num_states=100, seq_length=300, batch_size=512, num_epoch=10)

# load model
model_fname = f"/n/netscratch/sham_lab/Everyone/jchooi/in-context-language-learning/models/hmm-H-{hmm_args.num_states}-E-{hmm_args.num_emissions}-L-{hmm_args.seq_length}.pkl"
print(f"Loading model from {model_fname}")

model = torch.load(model_fname).to(device)

# load sample sentences
sample_sentences_filename = f"sentences-{hmm_args.num_emissions}.txt"
with open(sample_sentences_filename, 'r') as f:
    sample_sentences: List[str] = f.readlines()

#%%
# strip newlines
sample_sentences = [sentence.strip() for sentence in sample_sentences]

# concat into seq_len
text = ' '.join(sample_sentences)

# break into a batch
str_tokens = text.split(' ')
idx = 0
seqs: List[str] = []
while idx + hmm_args.seq_length < len(str_tokens):
    seq = str_tokens[idx:idx+hmm_args.seq_length]
    seqs.append(' '.join(seq))
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
emission_log_probs: Float[torch.Tensor, "batch seq_len n_emissions"] = model._emission_matrix(tokenized_sentences)
# get the argmax
print(f"{emission_log_probs.shape=}")
predicted_emissions = emission_log_probs.argmax(-1).cpu()
predicted_emissions_str = tokenizer.detokenize_batch(predicted_emissions)
for idx, sentence in enumerate(predicted_emissions_str):
    print("original sentence:")
    print(seqs[idx])
    print("predicted sentence:")
    print(sentence)
    print("-----"*5)
# probs are same across emissions, but they are different across words. This shouldn't happen, since otherwise some of the probabilities don't sum to 1.
# %%
# sum the emissions to see if wrong
# emission_probs = torch.exp(emission_log_probs).squeeze(0).sum(-1)
# print(emission_probs)
# %%
