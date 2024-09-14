#%%
"""Run this script with 'torchrun'."""
import os
import argparse
import signal
import contextlib
import gzip
import logging
from pathlib import Path
import sys
from typing import Optional, TextIO

import torch
import torch.multiprocessing as mp

from tqdm import tqdm
from olmo.beam_search import DeterministicSampler
from olmo.config import CheckpointType, TrainConfig
from olmo.exceptions import OLMoCliError, OLMoConfigurationError
from olmo.model import OLMo
from olmo.optim import BoltOnWarmupScheduler, build_optimizer, build_scheduler
from olmo.torch_util import peak_gpu_memory, seed_all
from olmo.train import Trainer
from olmo.util import clean_opt, prepare_cli_environment
from transformers import AutoTokenizer
from datasets import load_dataset

logger = logging.getLogger(__name__)
tokenizer_filepath = "/n/home07/jchooi/in-context-language-learning/27-sep/../olmo_data/tokenizers/tokenizer-1k"
load_path = "/n/holyscratch01/sham_lab/summer_2024/checkpoints/1k-vocab-1024-maxlength-100-components-46816664/step1000-unsharded"


try:
    mp.set_start_method("spawn", force=True)
except RuntimeError as e:
    print(f"failed to set multiprocessing start method: {e}")

prepare_cli_environment()

logger.info(f"multiprocessing start method set to '{mp.get_start_method()}'")

config_yaml = os.path.join(load_path,"config.yaml")

cfg = TrainConfig.load(config_yaml)

# Set CUDA device.
torch.cuda.set_device("cuda:0")
device = torch.device("cuda")

# Fill some configuration options.
cfg.model.precision = cfg.precision

# Set seed.
seed_all(cfg.seed)

# Initialize the model.
logger.info("Building model...")
olmo_model = OLMo(cfg.model)
logger.info(f"Total number of parameters: {olmo_model.num_params():,d}")
logger.info(f"Number of non-embedding parameters: {olmo_model.num_params(include_embedding=False):,d}")
logger.info(f"Peak GPU Memory (MB) after init: {int(peak_gpu_memory() or 0)}")
logger.info(f"Load path: {load_path}")

state_dict = torch.load(os.path.join(load_path,"model.pt")) 
olmo_model.load_state_dict(state_dict,assign=True)



olmo_model = olmo_model.to('cuda')
olmo_model.eval()
tokenizer = AutoTokenizer.from_pretrained(tokenizer_filepath)
# tokenizer.pad_token_id = tokenizer.eos_token_id
if tokenizer.pad_token_id is None:
    if tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

#%%
prompts = [
    """One day, a boy named Tim""",
    """One day, a boy named Tim went""",
    """One day, a boy named Tim went to""",
    """One day, a boy named Tim went to the""",
    """One day, a boy named Tim went to the park"""
]

raw_generations = generate_completions(
            olmo_model,
            tokenizer,
            prompts
        )

print(raw_generations)

for p in range(len(prompts)):
    print("input text:")
    print(prompts[p])

    print("output text:")
    print(raw_generations[p])
    print('\n')


#%%
def generate_completions(model, tokenizer, prompts):
    completions = []
    num_examples = len(prompts)
    batch_size = 1
    max_lengths = 1024
    assert num_examples%batch_size == 0
    for i in tqdm(range(0, num_examples, batch_size)):#tqdm(range(0, len(dataset), batch_size)):
        batch = prompts[i : i + batch_size]
        batch_tokens = tokenizer(
            batch, return_tensors="pt", padding="longest", add_special_tokens=True
        )
        batch_input_ids =  batch_tokens.input_ids 
        attention_mask = batch_tokens.attention_mask 
        input_tokens = batch_input_ids.shape[1]


        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        with torch.no_grad():
            batch_outputs = model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=attention_mask,
                    max_steps=max_lengths,
            ).token_ids.squeeze(1)
        
        generations = tokenizer.batch_decode(batch_outputs)
        completions += generations
  
    return completions
# %%
