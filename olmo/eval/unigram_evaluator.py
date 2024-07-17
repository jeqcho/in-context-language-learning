from typing import Any, Dict
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Metric
from .ngram_preprocess_batch import ngram_preprocess_batch
from .unigram_model import UnigramModel


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
        batch = ngram_preprocess_batch(batch)

        # train a bigram model
        inputs = batch["input_ids"]
        unigram_model = UnigramModel(dim=self.dim)

        # get the Q and P distribution for KL-divergence
        for i in range(len(inputs)):
            unigram_model.load(inputs[i])
            # get probabilities of the next-token prediction by bigram
            unigram_probs = unigram_model.get_probability_table()
            # focus on the first few relevant ones (non special tokens)
            current_logits = logits[i][-1][:self.dim]
            q = F.log_softmax(current_logits, dim=0)
            p = torch.tensor(unigram_probs).to(q.device)

            self.kl_divs.append(F.kl_div(q, p, reduction="sum"))
            unigram_model.reset()

    def compute(self) -> torch.Tensor:
        kl_div = torch.mean(torch.tensor(self.kl_divs))
        return kl_div
