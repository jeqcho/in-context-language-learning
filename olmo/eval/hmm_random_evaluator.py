from typing import Any, Dict
import logging
import torch
import torch.nn.functional as F
from torchmetrics import Metric
from .ngram_preprocess_batch import ngram_preprocess_batch
import numpy as np

log = logging.getLogger(__name__)


class KLHMMRandomMetric(Metric):
    full_state_update: bool = False
    metric_type = "kl-hmm-random-metric-type"

    def __init__(self, dim=10) -> None:
        super().__init__(sync_on_compute=True)

        self.dim = dim
        self.add_state("kl_divs", default=[], dist_reduce_fx=None)

    def reset(
        self,
    ):
        self.kl_divs = []

    def update(self, batch: Dict[str, Any], logits: torch.Tensor):
        # batch = ngram_preprocess_batch(batch)

        # get the Q and P distribution for KL-divergence
        ps = torch.tensor(
            np.stack([d["emission_probs"] for d in batch["metadata"]], axis=0)
        )
        ps = ps.to(logits.device)
        # current_logits = logits[:, -1, : self.dim]
        # use uniform logits as random
        current_logits = torch.full(size=(logits.shape[0],self.dim), fill_value=1/self.dim).to(logits.device)
        qs = F.log_softmax(current_logits, dim=1)
        self.kl_divs.append(F.kl_div(qs, ps, reduction="batchmean"))

    def compute(self) -> torch.Tensor:
        kl_div = torch.mean(torch.tensor(self.kl_divs))
        return kl_div
