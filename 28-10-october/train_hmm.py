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
    parser.add_argument('--num_components', type=int, 
                        help='Number of components in HMM')
    args = parser.parse_args()
    n_components = int(args.num_components)
    
    #%%
    maxlength = 438
    filename = "/n/holyscratch01/sham_lab/summer_2024/datasets/cleaned_tiny-600.npy"

    #%%
    # Load the memmap array
    data = np.load(filename)

    print("Done loading")
    
    # shortcut for testing
    # num_chosen_stories = 10
    num_chosen_stories = len(data)
    data = data[:num_chosen_stories]
    print(f"num_chosen_stories: {num_chosen_stories}")

    print(f"maxlength: {maxlength}")
    print(f"training for num states: {n_components}")
    #%%
    lengths = [maxlength] * len(data)
    # we also have to wrap each sequence with a [] since it is 1D
    # strip out the [] wrapping each story
    data = data.flatten()
    # wrap each token with [] as required in hmmlearn
    data = np.expand_dims(data,axis=1)
    print("Done wrapping")

    # %%
    start_time = time.time()
    model = hmm.CategoricalHMM(n_components=n_components).fit(data, lengths)
    with open(f"/n/holyscratch01/sham_lab/summer_2024/models/600-word-hmm-{n_components}.pkl", "wb") as file: pickle.dump(model, file)
    print(f"num_chosen_stories: {num_chosen_stories}")
    print(f"{n_components} component: %.2f seconds" % (time.time() - start_time))
    #%%

if __name__ == "__main__":
    main()
# %%
