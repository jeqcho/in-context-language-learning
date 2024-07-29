from typing import Any, Dict
import logging
import torch
import torch.nn.functional as F
from torchmetrics import Metric
import numpy as np
from .bigram_model import BigramModel, BatchedBigramModel
from .ngram_preprocess_batch import ngram_preprocess_batch

log = logging.getLogger(__name__)


class KLBigramMetric(Metric):
    # update method does not require access to global metric state
    full_state_update: bool = False
    metric_type = "kl-bigram-metric-type"

    def __init__(self, dim: int = 3) -> None:
        super().__init__(sync_on_compute=True)
        self.dim = dim
        self.add_state("kl_divs", default=[], dist_reduce_fx=None)

    def reset(
        self,
    ):
        self.kl_divs = []

    def update(self, batch: Dict[str, Any], logits: torch.Tensor):

        # batch = ngram_preprocess_batch(batch)

        inputs = batch["input_ids"]

        # train a bigram model
        batched_bigram_model = BatchedBigramModel(dim=self.dim)
        # get the Q and P distribution for KL-divergence
        ps = batched_bigram_model.load(inputs)
        # get all instances in the batch, last token, only consider first self.dim (vocab_size) logits
        current_logits = logits[:, -1, : self.dim]
        qs = F.log_softmax(current_logits, dim=1)
        self.kl_divs.append(F.kl_div(qs, ps, reduction="batchmean"))


    def compute(self) -> torch.Tensor:
        kl_div = torch.mean(torch.tensor(self.kl_divs))
        return kl_div
