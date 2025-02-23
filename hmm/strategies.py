"""
Calculate the logits under various strategies
"""

import torch
from jaxtyping import Int, Float
from hmm.HMMArgs import HMMArgs


def uniform(
    batch: Int[torch.Tensor, "batch seq_len dim"], hmm_args: HMMArgs
) -> Float[torch.Tensor, "batch num_emission"]:
    assert batch.ndim == 3
    assert batch[-1] == 1  # last dimension is 1d emissions

    batch_size, _, _ = batch.shape

    logits = torch.full(size=(batch_size, hmm_args.num_emissions), fill_value=1.0 / hmm_args.num_emissions)

    assert torch.allclose(logits.sum(dim=-1), torch.tensor(1.0))

    return logits


def unigram(
    batch: Int[torch.Tensor, "batch seq_len dim"], hmm_args: HMMArgs
) -> Float[torch.Tensor, "batch num_emission"]:
    assert batch.ndim == 3
    assert batch[-1] == 1  # last dimension is 1d emissions

    batch_size, _, _ = batch.shape

    # logits = torch.

    # assert torch.allclose(logits.sum(dim=-1), torch.tensor(1.0))

    # return logits
