"""
Helper classes for HMMs
"""

from dataclasses import dataclass


@dataclass
class HMMArgs:
    num_emissions: int
    num_states: int
    seq_length: int
    batch_size: int
    num_epoch: int