"""
Declares the HMMArgs dataclass
"""

from dataclasses import dataclass, field
from typing import Union, Literal


@dataclass
class HMMArgs:
    num_emissions: int
    num_states: int
    seq_length: int
    batch_size: int
    num_epoch: int
    model_filename: str = field(init=False)
    unique: bool
    update_freq: Union[int, Literal["all"]]  # how many batches before we call from_summaries()

    def __post_init__(self):
        self.model_filename = (
            f"/n/netscratch/sham_lab/Everyone/jchooi/in-context-language-learning/models/hmm-{self.__str__()}.pkl"
        )

    def __str__(self):
        string = f"H-{self.num_states}-E-{self.num_emissions}-L-{self.seq_length}-B-{self.batch_size}-update_freq-{self.update_freq}"
        if self.unique:
            string += "-unique"
        return string

    def epoch_stamped_filename(self, epoch: int) -> str:
        # remove .pkl
        truncated_filename = self.model_filename[:-4]

        truncated_filename += f"-epoch-{str(epoch)}.pkl"
        
        return truncated_filename

    def __hash__(self):
        return hash((self.num_emissions, self.num_states, self.seq_length, self.batch_size, self.num_epoch))
