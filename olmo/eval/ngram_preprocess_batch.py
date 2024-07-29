import numpy as np
from typing import Any, Dict
import logging

log = logging.getLogger(__name__)


def ngram_preprocess_batch(batch: Dict[str, Any]):
    assert "input_ids" in batch
    assert "metadata" in batch, "metadata required in batch for chosen_symbols but not found"

    inputs = batch["input_ids"]
    chosen_symbols = np.array([x['chosen_symbols'] for x in batch['metadata']])
    # log.info("chosen symbols")
    # log.info(chosen_symbols)
    # log.info("inputs")
    # log.info(inputs)
    # convert the inputs into [0, num_states)
    inputs_copy = inputs.clone().detach()
    for row_id in range(inputs.shape[0]):
        for idx, symbol in enumerate(chosen_symbols[row_id]):
            inputs_copy[row_id, inputs[row_id] == symbol] = idx
    batch["input_ids"] = inputs_copy.clone().detach()
    # log.info("final inputs")
    # log.info(batch["input_ids"])

    return batch
