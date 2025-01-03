from typing import Dict, List, Union

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchmetrics import MeanMetric, Metric

from olmo.eval.bigram_truth_evaluator import BigramTruthKLMetric
from olmo.eval.hmm_evaluator import KLHMMMetric
from olmo.eval.uniform_truth_evaluator import UniformTruthKLMetric
from olmo.eval.unigram_truth_evaluator import UnigramTruthKLMetric

from ..config import EvaluatorConfig, EvaluatorType, TrainConfig, CustomDataType
from ..exceptions import OLMoConfigurationError
from ..tokenizer import Tokenizer
from ..torch_util import get_global_rank, get_world_size
from .downstream import ICLMetric, label_to_task_map
from .evaluator import Evaluator

from .bigram_evaluator import KLBigramMetric
from .unigram_evaluator import KLUnigramMetric
from .uniform_evaluator import KLUniformMetric

__all__ = [
    "Evaluator",
    "ICLMetric",
    "label_to_task_map",
    "build_downstream_evaluator",
    "build_evaluator",
    "build_evaluators",
]


def build_downstream_evaluator(
    train_config: TrainConfig,
    eval_cfg: EvaluatorConfig,
    tokenizer: Tokenizer,
    device: torch.device,
    is_unit_test=False,
) -> Evaluator:
    task_kwargs = {}
    task_class = label_to_task_map[eval_cfg.label]
    if isinstance(task_class, tuple):
        task_class, task_kwargs = task_class
    ds_eval_dataset = task_class(tokenizer=tokenizer, **task_kwargs)  # type: ignore
    data_config = eval_cfg.data
    if is_unit_test:
        ds_eval_sampler = None
    else:
        ds_eval_sampler = DistributedSampler(
            ds_eval_dataset,
            drop_last=data_config.drop_last,
            shuffle=False,
            num_replicas=get_world_size(),
            rank=get_global_rank(),
            seed=train_config.seed,
        )
    ds_eval_dataloader = DataLoader(
        ds_eval_dataset,
        batch_size=eval_cfg.device_eval_batch_size or train_config.device_eval_batch_size,
        collate_fn=ds_eval_dataset.collate_fn,
        num_workers=data_config.num_workers,
        sampler=ds_eval_sampler,
        pin_memory=data_config.pin_memory,
        prefetch_factor=data_config.prefetch_factor,
        persistent_workers=data_config.persistent_workers,
        timeout=data_config.timeout,
    )
    metric = ICLMetric(metric_type=ds_eval_dataset.metric_type)

    evaluator = Evaluator(
        label=eval_cfg.label,
        type=eval_cfg.type,
        eval_loader=ds_eval_dataloader,
        eval_metric=metric.to(device),
        subset_num_batches=eval_cfg.subset_num_batches,
    )
    return evaluator


def build_evaluator(
    train_config: TrainConfig, eval_config: EvaluatorConfig, tokenizer: Tokenizer, device: torch.device
) -> Evaluator:
    from ..data import build_eval_dataloader, build_custom_dataloader

    if eval_config.data.use_train_custom_data_config:
        eval_config.data.custom_data_config = train_config.data.custom_data_config

    if eval_config.type == EvaluatorType.downstream:
        # Downstream evaluation.
        return build_downstream_evaluator(train_config, eval_config, tokenizer, device)
    elif eval_config.type == EvaluatorType.lm:
        # Language modeling evaluation.
        eval_loader = build_eval_dataloader(
            train_config,
            eval_config.data,
            eval_config.device_eval_batch_size or train_config.device_eval_batch_size,
        )

        def make_metric():
            return MeanMetric(nan_strategy="error").to(device)

        eval_metric: Union[Metric, Dict[str, Metric]]
        if eval_config.data.paths:
            eval_metric = make_metric()
        elif eval_config.data.datasets:
            eval_metric = {label: make_metric() for label in eval_config.data.datasets.keys()}
        else:
            raise OLMoConfigurationError("One of DataConfig.paths or DataConfig.datasets is required")

        return Evaluator(
            label=eval_config.label,
            type=eval_config.type,
            eval_loader=eval_loader,
            eval_metric=eval_metric,
            subset_num_batches=eval_config.subset_num_batches,
        )
    elif eval_config.type == EvaluatorType.bg:
        # KL divergence with bigram model
        # load the on-the-fly Markov chain generator
        assert train_config.data.custom_train_dataset
        eval_loader = build_custom_dataloader(train_config, eval_config.data)

        # use KL divergence
        def make_metric():
            if eval_config.data.custom_data_config.custom_data_type == CustomDataType.markov:
                return KLBigramMetric(dim=eval_config.data.custom_data_config.markov_dataset_config.num_states).to(
                    device
                )
            else:
                return KLBigramMetric(dim=eval_config.data.custom_data_config.hmm_dataset_config.num_symbols).to(
                    device
                )

        eval_metric: Union[Metric, Dict[str, Metric]]
        eval_metric = make_metric()

        # return an evaluator that computes the KL divergence with the bigram model
        return Evaluator(
            label=eval_config.label,
            type=eval_config.type,
            eval_loader=eval_loader,
            eval_metric=eval_metric,
            subset_num_batches=eval_config.subset_num_batches,
        )
    elif eval_config.type == EvaluatorType.uf:
        # KL divergence with uniform model
        # load the on-the-fly Markov chain generator
        assert train_config.data.custom_train_dataset
        eval_loader = build_custom_dataloader(train_config, eval_config.data)

        # use KL divergence
        def make_metric():
            if eval_config.data.custom_data_config.custom_data_type == CustomDataType.markov:
                return KLUniformMetric(dim=eval_config.data.custom_data_config.markov_dataset_config.num_states).to(
                    device
                )
            else:
                return KLUniformMetric(dim=eval_config.data.custom_data_config.hmm_dataset_config.num_symbols).to(
                    device
                )

        eval_metric: Union[Metric, Dict[str, Metric]]
        eval_metric = make_metric()

        # return an evaluator that computes the KL divergence with the bigram model
        return Evaluator(
            label=eval_config.label,
            type=eval_config.type,
            eval_loader=eval_loader,
            eval_metric=eval_metric,
            subset_num_batches=eval_config.subset_num_batches,
        )
    elif eval_config.type == EvaluatorType.ug:
        # KL divergence with unigram model
        # load the on-the-fly Markov chain generator
        assert train_config.data.custom_train_dataset
        eval_loader = build_custom_dataloader(train_config, eval_config.data)

        # use KL divergence
        def make_metric():
            if eval_config.data.custom_data_config.custom_data_type == CustomDataType.markov:
                return KLUnigramMetric(dim=eval_config.data.custom_data_config.markov_dataset_config.num_states).to(
                    device
                )
            else:
                return KLUnigramMetric(dim=eval_config.data.custom_data_config.hmm_dataset_config.num_symbols).to(
                    device
                )

        eval_metric: Union[Metric, Dict[str, Metric]]
        eval_metric = make_metric()

        # return an evaluator that computes the KL divergence with the bigram model
        return Evaluator(
            label=eval_config.label,
            type=eval_config.type,
            eval_loader=eval_loader,
            eval_metric=eval_metric,
            subset_num_batches=eval_config.subset_num_batches,
        )
    elif eval_config.type == EvaluatorType.hmm:
        # KL divergence with unigram model
        # load the on-the-fly Markov chain generator
        assert train_config.data.custom_train_dataset
        eval_loader = build_custom_dataloader(train_config, eval_config.data)

        # use KL divergence
        def make_metric():
            if eval_config.data.custom_data_config.custom_data_type == CustomDataType.markov:
                return KLHMMMetric(dim=eval_config.data.custom_data_config.markov_dataset_config.num_states).to(
                    device
                )
            else:
                return KLHMMMetric(dim=eval_config.data.custom_data_config.hmm_dataset_config.num_symbols).to(
                    device
                )

        eval_metric: Union[Metric, Dict[str, Metric]]
        eval_metric = make_metric()

        # return an evaluator that computes the KL divergence with the bigram model
        return Evaluator(
            label=eval_config.label,
            type=eval_config.type,
            eval_loader=eval_loader,
            eval_metric=eval_metric,
            subset_num_batches=eval_config.subset_num_batches,
        )
    elif eval_config.type == EvaluatorType.hmm_random:
        # KL divergence with unigram model
        # load the on-the-fly Markov chain generator
        assert train_config.data.custom_train_dataset
        eval_loader = build_custom_dataloader(train_config, eval_config.data)

        # use KL divergence
        def make_metric():
            if eval_config.data.custom_data_config.custom_data_type == CustomDataType.markov:
                return UniformTruthKLMetric(dim=eval_config.data.custom_data_config.markov_dataset_config.num_states).to(
                    device
                )
            else:
                return UniformTruthKLMetric(dim=eval_config.data.custom_data_config.hmm_dataset_config.num_symbols).to(
                    device
                )

        eval_metric: Union[Metric, Dict[str, Metric]]
        eval_metric = make_metric()

        # return an evaluator that computes the KL divergence with the bigram model
        return Evaluator(
            label=eval_config.label,
            type=eval_config.type,
            eval_loader=eval_loader,
            eval_metric=eval_metric,
            subset_num_batches=eval_config.subset_num_batches,
        )
    elif eval_config.type == EvaluatorType.hmm_bigram:
        # KL divergence with unigram model
        # load the on-the-fly Markov chain generator
        assert train_config.data.custom_train_dataset
        eval_loader = build_custom_dataloader(train_config, eval_config.data)

        # use KL divergence
        def make_metric():
            if eval_config.data.custom_data_config.custom_data_type == CustomDataType.markov:
                return BigramTruthKLMetric(dim=eval_config.data.custom_data_config.markov_dataset_config.num_states).to(
                    device
                )
            else:
                return BigramTruthKLMetric(dim=eval_config.data.custom_data_config.hmm_dataset_config.num_symbols).to(
                    device
                )

        eval_metric: Union[Metric, Dict[str, Metric]]
        eval_metric = make_metric()

        # return an evaluator that computes the KL divergence with the bigram model
        return Evaluator(
            label=eval_config.label,
            type=eval_config.type,
            eval_loader=eval_loader,
            eval_metric=eval_metric,
            subset_num_batches=eval_config.subset_num_batches,
        )
    elif eval_config.type == EvaluatorType.hmm_unigram:
        # KL divergence with unigram model
        # load the on-the-fly Markov chain generator
        assert train_config.data.custom_train_dataset
        eval_loader = build_custom_dataloader(train_config, eval_config.data)

        # use KL divergence
        def make_metric():
            if eval_config.data.custom_data_config.custom_data_type == CustomDataType.markov:
                return UnigramTruthKLMetric(dim=eval_config.data.custom_data_config.markov_dataset_config.num_states).to(
                    device
                )
            else:
                return UnigramTruthKLMetric(dim=eval_config.data.custom_data_config.hmm_dataset_config.num_symbols).to(
                    device
                )

        eval_metric: Union[Metric, Dict[str, Metric]]
        eval_metric = make_metric()

        # return an evaluator that computes the KL divergence with the bigram model
        return Evaluator(
            label=eval_config.label,
            type=eval_config.type,
            eval_loader=eval_loader,
            eval_metric=eval_metric,
            subset_num_batches=eval_config.subset_num_batches,
        )
    else:
        raise ValueError(f"Unexpected evaluator type '{eval_config.type}'")


def build_evaluators(cfg: TrainConfig, device: torch.device) -> List[Evaluator]:
    evaluators = []
    tokenizer = Tokenizer.from_train_config(cfg)
    for eval_cfg in cfg.evaluators:
        evaluators.append(build_evaluator(cfg, eval_cfg, tokenizer, device))
    return evaluators
