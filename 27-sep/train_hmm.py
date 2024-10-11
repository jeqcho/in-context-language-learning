#%%
from hmmlearn import hmm
import numpy as np
import time
import pickle
import argparse

#%%
def main():
    #%%
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Fit a HMM to token ids')
    parser.add_argument('--task_id', type=int, required=True, 
                        help='SLURM task ID')
    args = parser.parse_args()
    #%%
    list_of_n_components = [400, 500, 600, 700]
    # task ids are one-indexed
    n_components = list_of_n_components[args.task_id-1]
    
    #%%
    tokenizer_name = "tokenizer-828"
    split = "test"
    maxlength = 300
    total_tokens = 13815500 # Replace with the actual total number of tokens
    input_ids_path = f"/n/holyscratch01/sham_lab/summer_2024/datasets/tinystories-{split}-{tokenizer_name}-maxlength-{maxlength}/input_ids.npy"
    # shortcut, train on only the first 1000 stories
    rows = 5000
    print(f"tokenizer_name: {tokenizer_name}")
    print(f"split: {split}")
    print(f"maxlength: {maxlength}")
    print(f"total_tokens: {total_tokens}")
    print(f"total rows: {total_tokens/maxlength}")
    print(f"chosen rows: {rows}")
    # Load the .npy file

    # override total tokens
    total_tokens = rows * maxlength

    #%%
    # Load the memmap array
    data = np.memmap(
        str(input_ids_path), dtype=np.uint16, mode="r", shape=(total_tokens,)
    )
    print("Done loading")
    #%%
    lengths = [maxlength] * int(len(data)/maxlength)
    # we also have to wrap each sequence with a [] since it is 1D
    # wrap each number with []
    wrapped_data = np.array([[x] for x in data])
    print("Done wrapping")

    print(f"training for num states: {n_components}")

    # %%
    start_time = time.time()
    model = hmm.CategoricalHMM(n_components=n_components).fit(wrapped_data, lengths)
    with open(f"/n/holyscratch01/sham_lab/summer_2024/models/hmm-{n_components}-{split}-{tokenizer_name}-maxlength-{maxlength}-rows-{rows}.pkl", "wb") as file: pickle.dump(model, file)
    print(f"{n_components} component: %.2f seconds" % (time.time() - start_time))
    #%%

if __name__ == "__main__":
    main()
# %%
