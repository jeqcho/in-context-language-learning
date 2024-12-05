from typing import List, Dict
import re
import pickle


def explode_into_words(story: str) -> List[str]:
    # explode a story into a list of words and single-char non-alphanumerics (e.g. punctuations)
    return re.findall(r"\<\|endoftext\|\>|\w+|\W", story)


class custom_tokenizer:

    tokenizer_dict: Dict
    detokenize_dict: Dict
    eos_id: int

    def __init__(self, tokenizer_filename: str):
        with open(tokenizer_filename, "rb") as file:
            self.tokenizer_dict = pickle.load(file)

        # _tokenizer_dict is str word to int id dict
        eos_token = "<|endoftext|>"
        self.eos_id = self.tokenizer_dict[eos_token]

        self.detokenize_dict = dict()
        for k, v in self.tokenizer_dict.items():
            self.detokenize_dict[v] = k

    def tokenize_word(self, word: str):
        if word not in self.tokenizer_dict.keys():
            print(f"WARNING: {word} not in dict, using eos_id {self.eos_id} instead")
            return self.eos_id
        return self.tokenizer_dict[word]

    def detokenize_sentence(self, sentence: List[int], highlight: str = "") -> str:
        """Use the highlight to break apart tokens
        
        e.g. highlight ^ gives "The^boy^is^asleep^."
        if the words outputted doesn't have spaces.
        
        """
        return highlight.join([self.detokenize_dict[token_id] for token_id in sentence])

    def tokenize_sentences(self, sentences: List[str]) -> List[List[int]]:
        tokenized_sentences = []
        for sentence in sentences:
            tokenized_sentence = [self.tokenize_word(word.lower()) for word in explode_into_words(sentence)]
            tokenized_sentences.append(tokenized_sentence)
        return tokenized_sentences
