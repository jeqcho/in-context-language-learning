from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer
# Step 1: Load the dataset
data_files={"train": "TinyStoriesV2-GPT4-train.txt", "test": "TinyStoriesV2-GPT4-valid.txt"}
dataset = load_dataset('roneneldan/TinyStories',data_files=data_files, split='train')  # Replace 'dataset_name' and 'split' as needed

# Step 2: Load a tokenizer
tokenizer_name = "tinystories-1k"
tokenizer = AutoTokenizer.from_pretrained(f"../olmo_data/tokenizers/{tokenizer_name}")  # Replace with your model's tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Step 3: Define a tokenization function
def tokenize_function(examples):
    # Tokenize the 'text' column
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=1024)

# Step 4: Apply the tokenization function to the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Step 5: Extract the tokenized 'input_ids' into a NumPy array
input_ids = np.array(tokenized_dataset['input_ids'])

# Step 6: Save the arrays as .npy files
os.mkdir(f"/n/holyscratch01/sham_lab/summer_2024/datasets/{tokenizer_name}-length-1024")
np.save(f'/n/holyscratch01/sham_lab/summer_2024/datasets/{tokenizer_name}-length-1024/input_ids.npy', input_ids)