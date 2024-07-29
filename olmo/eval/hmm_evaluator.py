from typing import Any, Dict
import logging
import torch
import torch.nn.functional as F
from torchmetrics import Metric
from .ngram_preprocess_batch import ngram_preprocess_batch
import numpy as np

log = logging.getLogger(__name__)


class KLHMMMetric(Metric):
    full_state_update: bool = False
    metric_type = "kl-hmm-metric-type"

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
        last_tokens = batch["input_ids"][:, -1]
        next_emission_matrices = torch.tensor(
            np.stack([d["next_emission_matrix"] for d in batch["metadata"]], axis=0)
        )
        log.info(f"next_emission_matrices device {next_emission_matrices.device}")

        next_emission_matrices = next_emission_matrices.to(batch["input_ids"].device)

        log.info(f"next_emission_matrices device {next_emission_matrices.device}")

        ps = next_emission_matrices[torch.arange(batch["input_ids"].shape[0]), last_tokens]
        ps = torch.tensor(ps).to(logits.device)
        current_logits = logits[:, -1, : self.dim]
        qs = F.log_softmax(current_logits, dim=1)
        self.kl_divs.append(F.kl_div(qs, ps, reduction="batchmean"))

    def compute(self) -> torch.Tensor:
        kl_div = torch.mean(torch.tensor(self.kl_divs))
        return kl_div
