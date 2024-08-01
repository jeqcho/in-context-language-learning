from typing import Any, Dict
import logging
import torch
import torch.nn.functional as F
from torchmetrics import Metric

from olmo.eval.unigram_model import BatchedUnigramModel
import numpy as np

log = logging.getLogger(__name__)


class KLHMMUnigramMetric(Metric):
    full_state_update: bool = False
    metric_type = "kl-hmm-unigram-metric-type"

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
        # train a bigram model
        batched_unigram_model = BatchedUnigramModel(dim=self.dim)
        qs_exp = batched_unigram_model.load(batch["input_ids"].numpy())
        qs_exp = torch.tensor(qs_exp)
        qs = F.log_softmax(qs_exp, dim=1)
        self.kl_divs.append(F.kl_div(qs, ps, reduction="batchmean").item())

    def compute(self) -> torch.Tensor:
        kl_div = np.mean(self.kl_divs)
        return kl_div
