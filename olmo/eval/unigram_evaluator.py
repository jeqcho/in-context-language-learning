from typing import Any, Dict
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Metric
from .ngram_preprocess_batch import ngram_preprocess_batch
from .unigram_model import UnigramModel, BatchedUnigramModel


class KLUnigramMetric(Metric):
    full_state_update: bool = False
    metric_type = "kl-unigram-metric-type"

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
        device = "cpu"
        batch["input_ids"] = batch["input_ids"].to(device)
        logits = logits.to(device)

        # train a bigram model
        inputs:torch.Tensor = batch["input_ids"]

        # get the Q and P distribution for KL-divergence
        ps = []
        batched_unigram_model = BatchedUnigramModel(dim=self.dim)
        ps = batched_unigram_model.load(inputs.numpy())
        ps = torch.tensor(ps,device=device)
        # get all instances in the batch, last token, only consider first self.dim (vocab_size) logits
        current_logits = logits[:, -1, : self.dim]
        qs = F.log_softmax(current_logits, dim=1)
        self.kl_divs.append(F.kl_div(qs, ps, reduction="batchmean").item())


    def compute(self) -> torch.Tensor:
        kl_div = np.mean(self.kl_divs)
        return kl_div
