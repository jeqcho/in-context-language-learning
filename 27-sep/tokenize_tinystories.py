# %%
from datasets import load_dataset, Dataset
import numpy as np
from transformers import AutoTokenizer
import os
from pathlib import Path
from rich.progress import track

tokenizer_name = "tokenizer-828"
max_length = 500
split = "test"
# %%
# Step 1: Load the dataset
data_files={"train": "TinyStoriesV2-GPT4-train.txt", "test": "TinyStoriesV2-GPT4-valid.txt"}
dataset = load_dataset('roneneldan/TinyStories',data_files=data_files, split=split)  # Replace 'dataset_name' and 'split' as needed
print(dataset)

#%%
arr = ['']
eos = "<|endoftext|>\n"
for x in dataset['text']:
    if x == '':
        continue
    if arr[-1][-len(eos):] != eos:
        arr[-1] += x + '\n'
    elif x == "\n":
        continue
    else:
        arr.append(x + '\n')

#%%
# Convert the array to a dictionary with the key 'text'
data_dict = {"text": arr}

# Create a dataset from the dictionary
dataset = Dataset.from_dict(data_dict)
# Step 2: Load a tokenizer
# %%
tokenizer = AutoTokenizer.from_pretrained(f"../olmo_data/tokenizers/{tokenizer_name}")  # Replace with your model's tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Step 3: Define a tokenization function
num_actual_tokens = []
def tokenize_function(examples):
    # Tokenize the 'text' column
    global num_actual_tokens
    en = tokenizer(examples['text'])['input_ids']
    for tok in en:
        num_actual_tokens.append(len(tok))
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_length)

# Step 4: Apply the tokenization function to the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Calculating and printing statistics
num_actual_tokens = np.array(num_actual_tokens)
print(f"Mean: {np.mean(num_actual_tokens)}")
print(f"Median: {np.median(num_actual_tokens)}")
print(f"Q1 (25th Percentile): {np.percentile(num_actual_tokens, 25)}")
print(f"Q3 (75th Percentile): {np.percentile(num_actual_tokens, 75)}")
print(f"90th Percentile: {np.percentile(num_actual_tokens, 90)}")
print(f"99th Percentile: {np.percentile(num_actual_tokens, 99)}")
print(f"Max: {np.max(num_actual_tokens)}")

# Step 5: Extract the tokenized 'input_ids' into a NumPy array
print("Counting tokens...")
total_tokens = 0
mx = 0
for ex in track(tokenized_dataset):
    total_tokens += len(ex["input_ids"])  # type: ignore
    mx = max(mx,len(ex["input_ids"]) )
print(f"Max tokens: {mx}")
print(f"Total tokens: {total_tokens:,d}")

foldername = f"/n/holyscratch01/sham_lab/summer_2024/datasets/tinystories-{split}-{tokenizer_name}-maxlength-{max_length}"
print(f"Saving results to '{foldername}'...")
output_dir = Path(foldername)
output_dir.mkdir(exist_ok=True, parents=True)

input_ids_file = np.memmap(
    str(output_dir / "input_ids.npy"), dtype=np.uint16, mode="w+", shape=(total_tokens,)
)

# %%
offset = 0
for ex in track(tokenized_dataset):
    ex_len = len(ex["input_ids"])  # type: ignore
    input_ids_file[offset : offset + ex_len] = ex["input_ids"]  # type: ignore
    offset += ex_len
input_ids_file.flush()

print("Done!")
# %%
