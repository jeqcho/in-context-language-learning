#%%
from transformers import AutoTokenizer

vocab_size=1000
tokenizer_name="tokenizer-1k"
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
# %%

#%%
from datasets import load_dataset
data_files={"train": "TinyStoriesV2-GPT4-train.txt", "test": "TinyStoriesV2-GPT4-valid.txt"}
ds = load_dataset("roneneldan/TinyStories",data_files=data_files)
#%%

#%%
def get_training_corpus():
    return (
        ds["train"][i : i + 1000]['text']
        for i in range(0, len(ds["train"]), 1000)
    )


training_corpus = get_training_corpus()

#%%

tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, vocab_size=vocab_size)
# %%

#%%
example = """Sara and Ben are friends. They like to play with cards. They have many cards with animals and colors. They make piles and match the cards.\n\nOne day, Sara finds a card that is torn. It is a card with a big lion. She likes the lion card. She is sad. She shows the card to Ben.\n\n"Oh no, your card is broken!" Ben says. "Do you want to fix it?"\n\n"Yes, please. How can we fix it?" Sara asks.\n\nBen thinks. He has an idea. He gets some tape from his backpack. He puts the tape on the card. He sticks the two pieces together.\n\n"There, your card is fixed!" Ben says. He gives the card to Sara.\n\nSara smiles. She looks at the card. The lion is still there. The tape is shiny. She thinks the card is amazing.\n\n"Thank you, Ben. You are a good friend. You made my card amazing!" Sara says.\n\nBen smiles too. He is happy. He likes to help Sara. They hug. They play with the cards again. They have fun.'"""
#%%

#%%
tokens = old_tokenizer.tokenize(example)
tokens
# %%


#%%
tokens = tokenizer.tokenize(example)
tokens
#%%

tokenizer.save_pretrained(f"../olmo_data/tokenizers/{tokenizer_name}")
# %%
