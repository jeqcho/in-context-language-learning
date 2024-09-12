from hmmlearn import hmm
import numpy as np
import time
import pickle
import argparse

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Fit a HMM to token ids')
    parser.add_argument('--task_id', type=int, required=True, 
                        help='SLURM task ID')
    args = parser.parse_args()
    list_of_n_components = [100, 250, 500, 750, 1000, 2000]
    # task ids are one-indexed
    n_components = list_of_n_components[args.task_id-1]
    
    tokenizer_name = "tokenizer-1k"
    file_location = f"/n/holyscratch01/sham_lab/summer_2024/datasets/{tokenizer_name}-length-1024/input_ids.npy"
    # Load the .npy file
    data = np.load(file_location)

    # Display the contents of the file
    print("print(data)")
    print(data[0])
    print("data")
    print(data)

    lengths = [len(x) for x in data]
    # we also have to wrap each sequence with a [] since it is 1D
    # a list of numbers
    wrapped_data = np.concatenate(data)
    # wrap each number with []
    wrapped_data = np.array([[x] for x in wrapped_data])

    list_of_n_components = [100, 250, 500, 750, 1000]
    for n_components in list_of_n_components:
        start_time = time.time()
        model = hmm.CategoricalHMM(n_components=n_components).fit(wrapped_data, lengths)
        with open(f"hmm-{n_components}.pkl", "wb") as file: pickle.dump(model, file)
        print(f"{n_components} component: %.2f seconds" % (time.time() - start_time))

if __name__ == "__main__":
    main()