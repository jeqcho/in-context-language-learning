"""
This notebook loads a trained HMM and prompts it for text generation for manual sanity checks.

Input
- /n/netscratch/sham_lab/Everyone/jchooi/in-context-language-learning/models/hmm-H-{hmm_args.num_states}-E-{hmm_args.num_emissions}-L-{hmm_args.seq_length}.pkl
"""

# %%
from typing import List
from utils import *
import torch
from jaxtyping import Float

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hmm_args = HMMArgs(num_emissions=100, num_states=100, seq_length=100, batch_size=256, num_epoch=10)

# load model
print(f"Loading model from {hmm_args.model_filename}")

model = torch.load(hmm_args.model_filename).to(device)
hmm_wrapper = HMMWrapper(model, hmm_args)

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
emission_log_probs: Float[torch.Tensor, "batch seq_len n_hidden"] = model._emission_matrix(tokenized_sentences)

prediction_log_probs: Float[torch.Tensor, "batch seq_len n_hidden"] = model.predict_log_proba(tokenized_sentences)

#%%


#%%
# get the probabilities if we don't look at the whole sequence
# get the argmax
# predicted_states_no_peek = emission_log_probs.argmax(-1).cpu()

# # get the expected predicted state
# expected_predicted_states = hmm_wrapper.get_best_state_for_emission(tokenized_sentences)

# print(compare_tensors(predicted_states_no_peek, expected_predicted_states))

#%%
# get the predicted hidden states using all of the sequence
predicted_states = prediction_log_probs.argmax(-1).cpu()
print(f"{predicted_states.shape=}")
assert predicted_states.shape == (len(seqs), hmm_args.seq_length)

distributions = [hmm_wrapper.get_distributions_for_seq(seq) for seq in predicted_states]

predicted_emission_distributions = torch.tensor(distributions, dtype=torch.float)
print(f"{predicted_emission_distributions.shape=}")
assert predicted_emission_distributions.shape == (len(seqs), hmm_args.seq_length, hmm_args.num_emissions)

#%%
predicted_emissions = predicted_emission_distributions.argmax(-1)

predicted_emissions_str = tokenizer.detokenize_batch(predicted_emissions)
for idx, sentence in enumerate(predicted_emissions_str):
    compare_sentences(seqs[idx], sentence)
#%%