"""
Helper classes for HMMs
"""

from dataclasses import dataclass, field
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
    model_filename: str = field(init=False)
    
    def __post_init__(self):
        self.model_filename = f"/n/netscratch/sham_lab/Everyone/jchooi/in-context-language-learning/models/hmm-H-{self.num_states}-E-{self.num_emissions}-L-{self.seq_length}-epoch-{self.num_epoch}.pkl"
    

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
        self.num_hidden = len(model.distributions)
        self.num_emissions = model.distributions[0].probs.shape[1]

    # TODO
    def get_logits(
        self, tokenized_sentence: Int[torch.Tensor, "seq_len"]
    ) -> Float[torch.Tensor, "seq_len vocab_size"]:
        pass
    
    # get the emission distribution for each predicted hidden state
    def get_distributions_for_seq(self, seq):
        return [self.model.distributions[state].probs[0].tolist() for state in seq]
    
    def get_best_state_for_emission(self, emissions: Int[torch.Tensor, "batch seq_len"])->Int[torch.Tensor, "batch seq_len"]:
        """
        Returns the best hidden state for each emission state. Best here means the hidden state with the highest probability for that emission state across all other hidden states. Note that the emission state does not need to have the highest probability for that hidden state.
        """
        best_state_of = [None] * self.num_emissions
        for emission in range(self.num_emissions):
            mx = 0
            best_state = None
            for state in range(self.num_hidden):
                prob = self.model.distributions[state].probs[0][emission]
                if prob > mx:
                    mx = prob
                    best_state = state
            assert best_state is not None
            best_state_of[emission] = best_state
        if emissions is None:
            return torch.tensor(best_state_of)
        return emissions.apply_(lambda emission: best_state_of[emission])        
        
        


def compare_word_lists(word_list_1: List[str], word_list_2: List[str]) -> None:
    """
    Print out a nice comparison of the two word lists
    """
    assert len(word_list_1) == len(word_list_2)
    print(f"{'Sentence 1':<20} {'Sentence 2':<20}")
    print("=" * 40)
    for i in range(len(word_list_1)):
        word1 = word_list_1[i] if i < len(word_list_1) else ""
        word2 = word_list_2[i] if i < len(word_list_2) else ""
        print(f"{word1:<20} {word2:<20}")


def compare_sentences(sentence_1: str, sentence_2: str) -> None:
    """
    Print out a nice comparison of the two sentences
    """
    word_list_1 = sentence_1.split(" ")
    word_list_2 = sentence_2.split(" ")
    return compare_word_lists(word_list_1, word_list_2)


def compare_tensors(tensorA, tensorB):
    return torch.mean(tensorA == tensorB, dtype=torch.int)