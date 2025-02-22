"""
Helper classes for HMMs
"""

import json
import einops
import numpy as np
from pomegranate.distributions.categorical import Categorical
import torch
from jaxtyping import Int, Float
from typing import Any, Iterable, List, Tuple
from pomegranate.hmm import DenseHMM
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import wandb
from HMMArgs import HMMArgs
import time
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Tokenizer:
    def __init__(self, tokenizer_filename) -> None:
        with open(tokenizer_filename, "r") as f:
            self.tokenizer = json.load(f)

        self.detokenizer: List[str] = [None] * (max(self.tokenizer.values()) + 1)
        for string, token in self.tokenizer.items():
            self.detokenizer[token] = string

    def tokenize_sentence(self, str_sentence: str) -> Int[torch.Tensor, "seq_len"]:
        tokenized_sentence = [
            int(self.tokenizer[word]) for word in str_sentence.split(" ")
        ]
        return torch.tensor(tokenized_sentence, dtype=torch.int)

    def tokenize_batch(
        self, str_sentences: List[str]
    ) -> Int[torch.Tensor, "batch seq_len"]:
        batch_size = len(str_sentences)
        tokenized_sentences = []
        for i in range(batch_size):
            tokenized_sentence = [
                int(self.tokenizer[word]) for word in str_sentences[i].split(" ")
            ]
            tokenized_sentences.append(tokenized_sentence)
        return torch.tensor(tokenized_sentences, dtype=torch.int)

    def detokenize_sentence(
        self, tokenized_sentence: Int[torch.Tensor, "seq_len"]
    ) -> str:
        assert tokenized_sentence.device.type == "cpu"
        str_sentence = ""
        for token in tokenized_sentence:
            str_sentence += self.detokenizer[token] + " "
        return str_sentence.strip()

    def detokenize_batch(
        self, tokenized_sentences: Int[torch.Tensor, "seq_len"]
    ) -> List[str]:
        assert tokenized_sentences.device.type == "cpu"
        batch_size = tokenized_sentences.shape[0]
        str_sentences: List[str] = []
        for i in range(batch_size):
            str_sentence = ""
            for token in tokenized_sentences[i]:
                str_sentence += self.detokenizer[token] + " "
            str_sentences.append(str_sentence.strip())
        return str_sentences


class TimeTracker:
    start: float

    def __init__(self):
        self.start = time.time()

    def seconds_elapsed(self) -> int:
        """
        Returns time taken in integer seconds
        """
        now = time.time()
        elapsed_time = int(now - self.start)
        return elapsed_time

    def stop(self) -> str:
        """
        Returns time taken in zero-padded HH:MM:SS.SSS
        """
        return self.format_time(self.seconds_elapsed())

    @staticmethod
    def format_time(seconds: float) -> str:
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02}:{int(minutes):02}:{seconds:06.3f}"  # HH:MM:SS.SSS


class HMMWrapper:
    def __init__(self, model: DenseHMM, hmm_args: HMMArgs) -> None:
        self.model = model
        self.num_hidden = len(model.distributions)
        self.num_emissions = model.distributions[0].probs.shape[1]
        self.hmm_args = hmm_args
        self.tokens_seen = 0

    def __str__(self) -> str:
        return self.hmm_args.__str__()

    def get_distributions_for_seq(self, seq):
        """
        Get the emission distribution for each predicted hidden state
        """
        return [self.model.distributions[state].probs[0].tolist() for state in seq]

    def get_best_state_for_emission(
        self, emissions: Int[torch.Tensor, "batch seq_len"]
    ) -> Int[torch.Tensor, "batch seq_len"]:
        """
        Returns the best hidden state for each emission state.
        Best here means the hidden state with the highest probability for that emission state across all other hidden states.
        Note that the emission state does not need to have the highest probability for that hidden state.
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

    @torch.inference_mode()
    def get_final_token_cross_entropy(
        self, batch: Int[torch.Tensor, "batch seq_len emission_dim"]
    ) -> Float[torch.Tensor, "batch n_emissions"]:
        """
        Returns the mean of the cross entropy for the final token
        """
        # define parameters to help check shapes
        b, s, _ = batch.shape
        h = self.num_hidden
        e = self.num_emissions

        # we only have 1d emissions in our experiments
        assert batch.shape[-1] == 1

        # get the probabilites for the final hidden state
        hidden_state_prob = self.model.predict_proba(batch)
        assert hidden_state_prob.shape == (b, s, h)
        # check each row sums to one
        # print(f"max diff from 1 for hidden_state_prob: {abs(hidden_state_prob.sum(-1)-1.0).max()}")

        # get the transition matrix
        assert self.model.edges is not None
        transition_matrix = self.model.edges.exp()
        assert transition_matrix.shape == (h, h)
        # print(f"max diff from 1 for edges: {abs(transition_matrix.sum(-1)-1.0).max()}")

        # make them proper distributions
        hidden_state_prob = hidden_state_prob / hidden_state_prob.sum(-1, keepdim=True)
        transition_matrix = transition_matrix / transition_matrix.sum(-1, keepdim=True)

        assert torch.allclose(hidden_state_prob.sum(-1), torch.tensor(1.0), atol=1e-3)
        assert torch.allclose(transition_matrix.sum(-1), torch.tensor(1.0), atol=5e-2)

        # use the transition matrix to get a distribution for the next hidden state
        next_state_prob = einops.einsum(
            hidden_state_prob, transition_matrix, "b s h1, h1 h2 -> b s h2"
        )
        # print(f"max diff from 1 for next_state_prob: {abs(next_state_prob.sum(-1)-1.0).max()}")
        assert torch.allclose(next_state_prob.sum(-1), torch.tensor(1.0))

        # we only care about the last token (by taking the next token probability of the second to last token)
        next_state_prob = next_state_prob[:, -2, :]
        assert next_state_prob.shape == (b, h)

        # get the emisison matrix
        emission_matrix = torch.tensor(
            self.get_distributions_for_seq(range(self.num_hidden)), device=device
        )
        assert emission_matrix.shape == (h, e)
        # print(f"max diff from 1 for emission_matrix: {abs(emission_matrix.sum(-1)-1.0).max()}")
        assert torch.allclose(emission_matrix.sum(-1), torch.tensor(1.0))

        # use the emission matrix to get the probs for the next emission
        next_emission = einops.einsum(
            next_state_prob, emission_matrix, "b h, h e -> b e"
        )
        assert next_emission.shape == (b, e)
        # print(f"max diff from 1 for next_emission: {abs(next_emission.sum(-1)-1.0).max()}")
        assert torch.allclose(next_emission.sum(-1), torch.tensor(1.0))

        # get actual final tokens
        final_tokens = batch[:, -1, 0]
        assert final_tokens.shape == (b,)

        # print(f"san check {next_emission[0][0]=}")
        # print(f"san check {final_tokens[0]=}")

        # calculate the cross-entropy
        return torch.nn.functional.cross_entropy(next_emission, final_tokens)

    @torch.inference_mode()
    def get_final_token_statistics(self, unique=False) -> Tuple[float, float]:
        test_loader, total_len = get_test_loader(self.hmm_args, unique)
        pbar = tqdm(total=total_len)
        ce_list = []
        for batch, _ in test_loader:
            batch = batch.to(device)
            ce = self.get_final_token_cross_entropy(batch)
            ce_list.append(ce)
            pbar.update(batch.shape[0])
        pbar.close()
        ce_list = torch.tensor(ce_list)
        return ce_list.mean().item(), ce_list.std().item()

    def train(self, save_flag=True, save_freq=10, starting_epoch=0):
        # set up wandb
        wandb.init(project="in-context-language-learning", name=self.hmm_args.__str__())

        # train model
        train_loader, total_len = get_train_loader(self.hmm_args)
        print(
            f"Allocated memory after getting train loader: {torch.cuda.memory_allocated() / 1e9} GB"
        )
        print(
            f"Reserved memory after getting train loader: {torch.cuda.memory_reserved() / 1e9} GB"
        )
        dirty_flag = False
        for epoch_index in range(starting_epoch, self.hmm_args.num_epoch):
            print(f"Begin training for epoch {epoch_index+1}")
            epoch_time_tracker = TimeTracker()
            pbar = tqdm(total=total_len, desc=f"Epoch {epoch_index+1}")
            for idx, (batch, _) in enumerate(train_loader):
                batch = batch.to(device)
                self.tokens_seen += batch.numel()
                # model.fit(batch)
                self.model.summarize(batch)
                dirty_flag = True
                if self.hmm_args.update_freq != "all":
                    if idx % self.hmm_args.update_freq == 0:
                        self.model.from_summaries()
                        dirty_flag = False

                pbar.update(batch.shape[0])
            if dirty_flag:
                self.model.from_summaries()
            print(
                f"Allocated memory after this epoch: {torch.cuda.memory_allocated() / 1e9} GB"
            )
            print(
                f"Reserved memory after this epoch: {torch.cuda.memory_reserved() / 1e9} GB"
            )
            pbar.close()
            print(f"Training complete for epoch {epoch_index+1}!")
            print(f"Begin testing for epoch {epoch_index+1}")

            # testing
            print(f"Begin testing on duplicated sequences for epoch {epoch_index+1}")
            ce_loss, ce_std = self.get_final_token_statistics()
            print(f"Begin testing on unique sequences for epoch {epoch_index+1}")
            ce_loss_uniq, ce_std_uniq = self.get_final_token_statistics(unique=True)
            print(f"Testing complete for epoch {epoch_index+1}!")

            # save model
            if save_flag and ((epoch_index + 1) % save_freq == 0):
                print(f"Saving the model...")
                save_time_tracker = TimeTracker()
                torch.save(
                    self.model, self.hmm_args.epoch_stamped_filename(epoch_index + 1)
                )
                print(f"Time taken to save model: {save_time_tracker.stop()}")
                print(f"Model saved!")

            # logging
            wandb.log(
                {
                    "test-loss": ce_loss,
                    "test-std": ce_std,
                    "test-loss-unique": ce_loss_uniq,
                    "test-std-unique": ce_std_uniq,
                    "tokens-seen": self.tokens_seen,
                    "time-epoch-s": epoch_time_tracker.seconds_elapsed(),
                    "epoch": epoch_index + 1,
                },
                step=epoch_index,
                commit=True,
            )
            print(f"Time taken for epoch {epoch_index+1}: {epoch_time_tracker.stop()}")
            print(f"Epoch {epoch_index+1} is complete!")

        wandb.finish()


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


def get_test_loader(hmm_args: HMMArgs, unique: bool = False) -> Tuple[Iterable, int]:
    test_fname = f"/n/netscratch/sham_lab/Everyone/jchooi/in-context-language-learning/data/TinyStories-{hmm_args.num_emissions}-test{'-unique' if unique else ''}.txt"

    with open(test_fname, "r") as f:
        test_lines = f.readlines()

    # concat into a big string and split into seq_length
    test_lines = [line.strip() for line in test_lines]
    test_string = " ".join(test_lines)
    test_integers = [int(token) for token in test_string.split(" ")]

    # log GPU
    print(f"Allocated memory before: {torch.cuda.memory_allocated() / 1e9} GB")
    print(f"Reserved memory before: {torch.cuda.memory_reserved() / 1e9} GB")

    # remove trailing sequence
    extra_length = len(test_integers) % hmm_args.seq_length
    if extra_length > 0:
        train_integers = test_integers[:-extra_length]
    else:
        train_integers = test_integers

    train_array = torch.tensor(train_integers).reshape(-1, hmm_args.seq_length)

    # wrap each emission as 1d
    train_array = torch.unsqueeze(train_array, -1)

    train_dataset = TensorDataset(
        train_array, torch.empty_like(train_array)
    )  # labels are dummy tensors
    return DataLoader(train_dataset, batch_size=hmm_args.batch_size, shuffle=True), len(
        train_dataset
    )


def get_train_loader(hmm_args: HMMArgs) -> Tuple[Iterable, int]:
    # get training data
    train_fname = f"/n/netscratch/sham_lab/Everyone/jchooi/in-context-language-learning/data/TinyStories-{hmm_args.num_emissions}-train{'-unique' if hmm_args.unique else ''}.txt"

    with open(train_fname, "r") as f:
        train_lines = f.readlines()

    # concat into a big string and split into seq_length
    train_lines = [line.strip() for line in train_lines]
    train_string = " ".join(train_lines)
    train_integers = [int(token) for token in train_string.split(" ")]

    print(f"Total number of training tokens: {len(train_integers)}")

    # log GPU
    print(f"Allocated memory before: {torch.cuda.memory_allocated() / 1e9} GB")
    print(f"Reserved memory before: {torch.cuda.memory_reserved() / 1e9} GB")

    # remove trailing sequence
    extra_length = len(train_integers) % hmm_args.seq_length
    train_integers = train_integers[:-extra_length]
    train_array = torch.tensor(train_integers).reshape(-1, hmm_args.seq_length)

    # wrap each emission as 1d
    train_array = torch.unsqueeze(train_array, -1)

    train_dataset = TensorDataset(
        train_array, torch.empty_like(train_array)
    )  # labels are dummy tensors

    return DataLoader(train_dataset, batch_size=hmm_args.batch_size, shuffle=True), len(
        train_dataset
    )


def init_model(hmm_args: HMMArgs) -> DenseHMM:
    hidden_states: List[Any] = [None] * hmm_args.num_states
    rng = np.random.default_rng(42)
    for i in range(hmm_args.num_states):
        dist = rng.uniform(low=0, high=1, size=(1, hmm_args.num_emissions))
        dist = dist / dist.sum()
        hidden_states[i] = Categorical(torch.tensor(dist).tolist())
    edges = torch.full(
        size=(hmm_args.num_states, hmm_args.num_states),
        fill_value=1.0 / hmm_args.num_states,
    ).tolist()
    starts = torch.full(
        size=(hmm_args.num_states,), fill_value=1.0 / hmm_args.num_states
    ).tolist()
    ends = torch.full(
        size=(hmm_args.num_states,), fill_value=1.0 / hmm_args.num_states
    ).tolist()

    model = DenseHMM(
        hidden_states, edges=edges, starts=starts, ends=ends, verbose=False
    )
    return model


def load_model(hmm_args: HMMArgs, epoch_on_filename: int) -> DenseHMM:
    model = torch.load(hmm_args.epoch_stamped_filename(epoch_on_filename))
    return model


def get_hmm_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a Hidden Markov Model (HMM) using the pomegranate library."
    )
    parser.add_argument(
        "--num_emissions",
        type=int,
        required=True,
        help="Number of emissions in the HMM",
    )
    parser.add_argument(
        "--num_states", type=int, required=True, help="Number of states in the HMM"
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        required=True,
        help="Length of the sequences used for training",
    )
    parser.add_argument(
        "--batch_size", type=int, required=True, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epoch", type=int, required=True, help="Number of epochs for training"
    )
    parser.add_argument(
        "--update_freq",
        type=int,
        help="Number of batches before calling from_summaries.",
    )
    parser.add_argument(
        "--unique", action="store_true", help="Train on unique sentences only"
    )
    parser.add_argument(
        "--save_epoch_freq",
        type=int,
        help="Save the model at that epoch frequency. If save_epoch_freq=5, save model after 5 epochs of training.",
    )
    return parser
