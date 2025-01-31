"""
Helper classes for HMMs
"""

from dataclasses import dataclass, field
import json
import einops
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
    
    def __str__(self):
        return f"HMMArgs(num_emissions={self.num_emissions}, num_states={self.num_states}, seq_length={self.seq_length}, batch_size={self.batch_size}, num_epoch={self.num_epoch})"
    

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
    
    def get_final_token_cross_entropy(self, batch: Int[torch.Tensor, "batch seq_len emission_dim"]) -> Float[torch.Tensor, "batch n_emissions"]:
        """
        Returns the mean of the cross entropy for the final token
        """
        # define parameters to help check shapes
        b, s, _ = batch.shape
        h = self.model.n_distributions
        
        # get the probabilites for the final hidden state
        hidden_state_prob = self.model.predict_proba(batch)
        assert hidden_state_prob.shape == (b, s, h)
        # check each row sums to one
        print(f"max diff from 1 for hidden_state_prob: {abs(hidden_state_prob.sum(-1)-1.0).max()}")
        assert torch.allclose(hidden_state_prob.sum(-1), torch.tensor(1.0), atol=1e-5)
        
        # get the transition matrix
        assert self.model.edges is not None
        transition_matrix = self.model.edges.exp()
        assert transition_matrix.shape == (h, h)
        print(f"max diff from 1 for edges: {abs(transition_matrix.sum(-1)-1.0).max()}")
        assert torch.allclose(transition_matrix.sum(-1), torch.tensor(1.0), atol=5e-2)
        
        # make them proper distributions
        hidden_state_prob = hidden_state_prob / hidden_state_prob.sum(-1, keepdim=True)
        transition_matrix = transition_matrix / transition_matrix.sum(-1, keepdim=True)
        
        # use the transition matrix to get a distribution for the next hidden state
        next_state_prob = einops.einsum(hidden_state_prob, transition_matrix, "b s h1, h1 h2 -> b s h2")
        print(f"max diff from 1 for next_state_prob: {abs(next_state_prob.sum(-1)-1.0).max()}")
        assert torch.allclose(next_state_prob.sum(-1), torch.tensor(1.0))
        
        # get the emisison matrix
        
        # use the emission matrix to get the probs for the next emission
        
        
        


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