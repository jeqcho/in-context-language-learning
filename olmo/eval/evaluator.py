from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric, Metric

from olmo.eval.bigram_truth_evaluator import BigramTruthKLMetric
from olmo.eval.uniform_truth_evaluator import UniformTruthKLMetric
from olmo.eval.unigram_truth_evaluator import UnigramTruthKLMetric

from ..config import EvaluatorType
from .downstream import ICLMetric
from .bigram_evaluator import KLBigramMetric
from .unigram_evaluator import KLUnigramMetric
from .uniform_evaluator import KLUniformMetric
from .hmm_evaluator import KLHMMMetric

__all__ = ["Evaluator"]


@dataclass
class Evaluator:
    label: str
    type: EvaluatorType
    eval_loader: DataLoader
    eval_metric: Union[Metric, Dict[str, Metric]]
    subset_num_batches: Optional[int] = None

    def reset_metrics(self) -> None:
        if isinstance(self.eval_metric, Metric):
            self.eval_metric.reset()
        else:
            for metric in self.eval_metric.values():
                metric.reset()

    def compute_metrics(self) -> Dict[str, float]:
        if self.type == EvaluatorType.downstream:
            assert isinstance(self.eval_metric, ICLMetric)
            value = self.eval_metric.compute().item()
            key = f"eval/downstream/{self.label}_{self.eval_metric.metric_type}"
            if self.eval_metric.metric_type == "ce_loss":
                key = key.replace("/downstream/", "/downstream_ce_loss/")
            return {key: value}
        elif self.type == EvaluatorType.lm:
            # Metric(s) = cross entropy loss
            metrics: Dict[str, Metric]
            if isinstance(self.eval_metric, Metric):
                metrics = {self.label: self.eval_metric}
            else:
                metrics = self.eval_metric
            out = {}
            for label in sorted(metrics.keys()):
                metric = metrics[label]
                assert isinstance(metric, MeanMetric)
                if metric.weight.item() == 0.0:  # type: ignore
                    # In this case we probably haven't called '.update()' on this metric yet,
                    # so we do so here with dummy values. Since we pass 0.0 in for weight this won't
                    # affect the final value.
                    # This can happen when the evaluator contains multiple tasks/datasets and we didn't
                    # get to this one within the current evaluation loop.
                    metric.update(0.0, 0.0)
                loss = metric.compute()
                if loss.isnan().item():
                    # This can happen when the evaluator contains multiple tasks/datasets and we didn't
                    # get to this one within the current evaluation loop.
                    continue
                else:
                    out[f"eval/{label}/CrossEntropyLoss"] = loss.item()
                    out[f"eval/{label}/Perplexity"] = torch.exp(loss).item()
            return out
        elif self.type == EvaluatorType.bg:
            # bigram
            # raise Exception("Possible bug in bigram implementation")
            assert isinstance(self.eval_metric, KLBigramMetric)
            value = self.eval_metric.compute().item()
            key = f"eval/{self.label}_{self.eval_metric.metric_type}"
            return {key: value}
        elif self.type == EvaluatorType.ug:
            # unigram
            # raise Exception("Possible bug in unigram implementation")
            assert isinstance(self.eval_metric, KLUnigramMetric)
            value = self.eval_metric.compute().item()
            key = f"eval/{self.label}_{self.eval_metric.metric_type}"
            return {key: value}
        elif self.type == EvaluatorType.uf:
            assert isinstance(self.eval_metric, KLUniformMetric)
            value = self.eval_metric.compute().item()
            key = f"eval/{self.label}_{self.eval_metric.metric_type}"
            return {key: value}
        elif self.type == EvaluatorType.hmm:
            assert isinstance(self.eval_metric, KLHMMMetric)
            value = self.eval_metric.compute().item()
            key = f"eval/{self.label}_{self.eval_metric.metric_type}"
            return {key: value} # tbh the code is the same for all these metrics
        elif self.type == EvaluatorType.hmm_random:
            assert isinstance(self.eval_metric, UniformTruthKLMetric)
            value = self.eval_metric.compute().item()
            key = f"eval/{self.label}_{self.eval_metric.metric_type}"
            return {key: value} # tbh the code is the same for all these metrics
        elif self.type == EvaluatorType.hmm_bigram:
            assert isinstance(self.eval_metric, BigramTruthKLMetric)
            value = self.eval_metric.compute().item()
            key = f"eval/{self.label}_{self.eval_metric.metric_type}"
            return {key: value} # tbh the code is the same for all these metrics
        elif self.type == EvaluatorType.hmm_unigram:
            assert isinstance(self.eval_metric, UnigramTruthKLMetric)
            value = self.eval_metric.compute().item()
            key = f"eval/{self.label}_{self.eval_metric.metric_type}"
            return {key: value} # tbh the code is the same for all these metrics
        else:
            raise ValueError(f"Unexpected evaluator type '{self.type}'")

    def update_metrics(
        self,
        batch: Dict[str, Any],
        ce_loss: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        if self.type == EvaluatorType.downstream:
            assert isinstance(self.eval_metric, ICLMetric)
            self.eval_metric.update(batch, logits)  # type: ignore
        elif self.type == EvaluatorType.lm:
            # Metric(s) = cross entropy loss
            for metadata, instance_loss in zip(batch["metadata"], ce_loss):
                if isinstance(self.eval_metric, dict):
                    metric = self.eval_metric[metadata["label"]]
                else:
                    metric = self.eval_metric
                metric.update(instance_loss)
        elif self.type == EvaluatorType.bg:
            # bigram
            assert isinstance(self.eval_metric, KLBigramMetric)
            self.eval_metric.update(batch, logits)
        elif self.type == EvaluatorType.ug:
            # unigram
            assert isinstance(self.eval_metric, KLUnigramMetric)
            self.eval_metric.update(batch, logits)
        elif self.type == EvaluatorType.uf:
            # uniform
            assert isinstance(self.eval_metric, KLUniformMetric)
            self.eval_metric.update(batch, logits)
        elif self.type == EvaluatorType.hmm:
            # hmm ground truth
            assert isinstance(self.eval_metric, KLHMMMetric)
            self.eval_metric.update(batch, logits)
        elif self.type == EvaluatorType.hmm_random:
            # hmm ground truth
            assert isinstance(self.eval_metric, UniformTruthKLMetric)
            self.eval_metric.update(batch, logits)
        elif self.type == EvaluatorType.hmm_bigram:
            # hmm ground truth
            assert isinstance(self.eval_metric, BigramTruthKLMetric)
            self.eval_metric.update(batch, logits)
        elif self.type == EvaluatorType.hmm_unigram:
            # hmm ground truth
            assert isinstance(self.eval_metric, UnigramTruthKLMetric)
            self.eval_metric.update(batch, logits)
        else:
            raise ValueError(f"Unexpected evaluator type '{self.type}'")
