"""
Helper classes for HMMs
"""

from dataclasses import dataclass
import json
import torch
from jaxtyping import Int, Float
from typing import List
from pomegranate.hmm import DenseHMM


@dataclass
class HMMArgs:
    num_emissions: int
    num_states: int
    seq_length: int
    batch_size: int
    num_epoch: int


class Tokenizer:
    def __init__(self, tokenizer_filename) -> None:
        with open(tokenizer_filename, "r") as f:
            self.tokenizer = json.load(f)

        self.detokenizer: List[str] = [None] * (max(self.tokenizer.values()) + 1)
        for string, token in self.tokenizer.items():
            self.detokenizer[token] = string

    def tokenize_sentence(self, str_sentence: str) -> Int[torch.Tensor, "seq_len"]:
        tokenized_sentence = [int(self.tokenizer[word]) for word in str_sentence.split(" ")]
        return torch.tensor(tokenized_sentence, dtype=torch.int)

    def tokenize_batch(self, str_sentences: List[str]) -> Int[torch.Tensor, "batch seq_len"]:
        batch_size = len(str_sentences)
        tokenized_sentences = []
        for i in range(batch_size):
            tokenized_sentence = [int(self.tokenizer[word]) for word in str_sentences[i].split(" ")]
            tokenized_sentences.append(tokenized_sentence)
        return torch.tensor(tokenized_sentences, dtype=torch.int)

    def detokenize_sentence(self, tokenized_sentence: Int[torch.Tensor, "seq_len"]) -> str:
        assert tokenized_sentence.device.type == "cpu"
        str_sentence = ""
        for token in tokenized_sentence:
            str_sentence += self.detokenizer[token] + " "
        return str_sentence.strip()

    def detokenize_batch(self, tokenized_sentences: Int[torch.Tensor, "seq_len"]) -> List[str]:
        assert tokenized_sentences.device.type == "cpu"
        batch_size = tokenized_sentences.shape[0]
        str_sentences: List[str] = []
        for i in range(batch_size):
            str_sentence = ""
            for token in tokenized_sentences[i]:
                str_sentence += self.detokenizer[token] + " "
            str_sentences.append(str_sentence.strip())
        return str_sentences


class HMMWrapper:
    def __init__(self, model: DenseHMM) -> None:
        self.model = model
    
    # TODO
    def get_logits(self, tokenized_sentence: Int[torch.Tensor, "seq_len"]) -> Float[torch.Tensor, "seq_len vocab_size"]:
        pass