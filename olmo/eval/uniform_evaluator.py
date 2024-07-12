from typing import Any, Dict

import torch
import torch.nn.functional as F
from torchmetrics import Metric


class KLUniformMetric(Metric):
    full_state_update: bool = False
    metric_type = "kl-uniform-metric-type"

    def __init__(self, dim=10) -> None:
        super().__init__(sync_on_compute=True)

        self.dim = 3
        self.add_state("kl_divs", default=[], dist_reduce_fx=None)

    def reset(
        self,
    ):
        self.kl_divs = []

    def update(self, batch: Dict[str, Any], logits: torch.Tensor):
        # save the batch info (inputs and logits)
        # temp fix for key name in batch
        if "input_ids" not in batch:
            assert "inputs" in batch
            batch["input_ids"] = batch["inputs"]

        # train a bigram model
        inputs = batch["input_ids"]

        # get the Q and P distribution for KL-divergence
        for i in range(len(inputs)):
            current_logits = logits[i][-1][:self.dim]
            q = F.log_softmax(current_logits, dim=0)
            p = torch.tensor([1/self.dim]*self.dim).to(q.device)
            self.kl_divs.append(F.kl_div(q, p, reduction="sum"))

    def compute(self) -> torch.Tensor:
        kl_div = torch.mean(torch.tensor(self.kl_divs))
        return kl_div
