from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer
# Step 1: Load the dataset

dataset = load_dataset('noanabeshima/TinyStoriesV2', split='validation')  # Replace 'dataset_name' and 'split' as needed

# Step 2: Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')  # Replace with your model's tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Step 3: Define a tokenization function
def tokenize_function(examples):
    # Tokenize the 'text' column
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

# Step 4: Apply the tokenization function to the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Step 5: Extract the tokenized 'input_ids' into a NumPy array
input_ids = np.array(tokenized_dataset['input_ids'])

# Step 6: Save the arrays as .npy files
np.save('/n/holyscratch01/sham_lab/summer_2024/datasets/tinystories/input_ids.npy', input_ids)