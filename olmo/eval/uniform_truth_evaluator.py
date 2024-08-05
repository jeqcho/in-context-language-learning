from typing import Any, Dict
import logging
import torch
import torch.nn.functional as F
from torchmetrics import Metric
from .ngram_preprocess_batch import ngram_preprocess_batch
import numpy as np

log = logging.getLogger(__name__)


class UniformTruthKLMetric(Metric):
    full_state_update: bool = False
    metric_type = "kl-uniform-vs-truth"

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

        # since the GPU is running out of memory
        # run these operations in CPU instead
        self.original_device = batch["input_ids"].device
        device = "cpu"
        batch["input_ids"] = batch["input_ids"].to(device)
        logits = logits.to(device)

        # get the Q and P distribution for KL-divergence
        ps = torch.tensor(np.stack([d["emission_probs"] for d in batch["metadata"]], axis=0), device=logits.device)
        # current_logits = logits[:, -1, : self.dim]
        # use uniform logits as random
        current_logits = torch.full(
            size=(logits.shape[0], self.dim), fill_value=1 / self.dim, device=logits.device
        )
        qs = F.log_softmax(current_logits, dim=1)
        self.kl_divs.append(F.kl_div(qs, ps, reduction="batchmean").item())

    def compute(self) -> torch.Tensor:
        kl_div = np.mean(self.kl_divs)
        return kl_div
