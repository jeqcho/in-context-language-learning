"""
Uses HMMs to generate synthetic data for LLMs.
"""

# %%
import os
from DataGenerator import DataGenerator
from hmm.utils import load_model, HMMWrapper, TimeTracker, get_hmm_args_parser
from hmm.HMMArgs import HMMArgs

from numpy.typing import NDArray
import numpy as np

if __name__ == "__main__":
    parser = get_hmm_args_parser()
    parser.add_argument(
        "--load_model_with_epoch", type=int, required=True, help="Load the model from file."
    )
    parser.add_argument("--permutate_emissions", action="store_true", help="Permutate the emission matrix.")
    parser.add_argument(
        "--gen_seq_len", type=int, required=True, help="Sequence length to generate for each sequence."
    )
    parser.add_argument(
        "--num_seq", type=int, required=True, help="Number of sequences to generate."
    )
    parser.add_argument(
        "--suffix", type=str, help="Optional suffix (e.g. test)"
    )
    parser.add_argument(
        "--gen_batch_size", type=int, required=True, help="Batch size when generating sequences."
    )
    args = parser.parse_args()
    
    args.permutate_emissions = bool(args.permutate_emissions)
    print(args)
    
    # init params
    hmm_args = HMMArgs(
        num_emissions=args.num_emissions,
        num_states=args.num_states,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        num_epoch=args.num_epoch,
        unique=args.unique,
        update_freq=args.update_freq,
    )

    # track time
    tracker = TimeTracker()

    # load model
    model = load_model(hmm_args, epoch_on_filename=args.load_model_with_epoch)
    hmm_wrapper = HMMWrapper(model, hmm_args)

    data_generator = DataGenerator(
        gen_seq_len=args.gen_seq_len,
        num_seq=args.num_seq,
        permutate_emissions=args.permutate_emissions,
        hmm_wrapper=hmm_wrapper,
        epoch_on_filename=args.load_model_with_epoch,
        suffix=args.suffix
    )

    # %%
    print("Generating data...")
    result: NDArray[np.int_] = data_generator.generate_all(batch_size=args.gen_batch_size)
    print("Data generation complete!")

    print(f"Saving data to {data_generator.data_filename}")
    np.save(data_generator.data_filename, result)
    print("Data saved!")
    
    #  Clean up the temporary file
    temp_filename = f"{data_generator.data_filename}.tmp"
    if os.path.exists(temp_filename):
        os.remove(temp_filename)

    # log time
    print(f"Total time taken: {tracker.format_time(tracker.seconds_elapsed())}")
    # %%
