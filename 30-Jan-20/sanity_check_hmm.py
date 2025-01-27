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
import torch as t
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import json

# %%
device = t.device("cuda" if t.cuda.is_available() else "cpu")

if __name__ == "__main__":
    hmm_args = HMMArgs(num_emissions=100, num_states=100, seq_length=100, batch_size=512, num_epoch=10)
    
    # load model
    model_fname = f"/n/netscratch/sham_lab/Everyone/jchooi/in-context-language-learning/models/hmm-H-{hmm_args.num_states}-E-{hmm_args.num_emissions}-L-{hmm_args.seq_length}.pkl"

    model = t.load(model_fname).to(device)
    
    # load sample sentences
    sample_sentences_filename = "sentences.txt"
    with open(sample_sentences_filename, 'r') as f:
        sample_sentences: List[str] = f.readlines()
    
    # tokenize
    tokenizer_filename = "/n/netscratch/sham_lab/Everyone/jchooi/in-context-language-learning/data/tokenizer.json"
    tokenizer = Tokenizer(tokenizer_filename)
    chosen_sentence = sample_sentences[0]
    tokenized_sentence = tokenizer.tokenize_sentence(chosen_sentence)
    
    # pass it to the HMM
    
    

# %%
