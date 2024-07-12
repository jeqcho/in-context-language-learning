from typing import Any, Dict
import logging
import torch
import torch.nn.functional as F
from torchmetrics import Metric
import numpy as np
from .bigram_model import BigramModel

log = logging.getLogger(__name__)

class KLBigramMetric(Metric):
    # update method does not require access to global metric state
    full_state_update: bool = False
    metric_type = "kl-bigram-metric-type"

    def __init__(self, dim=10) -> None:
        super().__init__(sync_on_compute=True)

        self.dim = 3
        self.add_state("kl_divs", default=[], dist_reduce_fx=None)

    def reset(
        self,
    ):
        self.kl_divs = []

    def update(self, batch: Dict[str, Any], logits: torch.Tensor):
        # temp fix for key name in batch
        if "input_ids" not in batch:
            assert "inputs" in batch
            batch["input_ids"] = batch["inputs"]

        inputs = batch["input_ids"]

        # train a bigram model
        bigram_model = BigramModel(dim=self.dim)

        # get the Q and P distribution for KL-divergence
        for i in range(len(inputs)):
            for j in range(1, len(inputs[i])):
                bigram_model.update(inputs[i][j - 1], inputs[i][j])
            # get probabilities of the next-token prediction by bigram
            bigram_probs = bigram_model.get_transition_matrix()[inputs[i][-1]]
            current_logits = logits[i][-1][:self.dim]
            q = F.log_softmax(current_logits, dim=0)
            p = torch.tensor(bigram_probs).to(q.device)
            # reset the model for next Markov chain instance
            self.kl_divs.append(F.kl_div(q, p, reduction="sum"))
            bigram_model.reset()

    def compute(self) -> torch.Tensor:
        kl_div = torch.mean(torch.tensor(self.kl_divs))
        return kl_div

