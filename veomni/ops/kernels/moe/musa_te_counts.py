"""Guarded integer column counts for the qualified MUSA E32 routing shapes."""

from collections import Counter

import torch


COUNTERS = Counter()


def column_counts(mask, hidden, probabilities, experts):
    """Use an integer reduction for the long Qwen3.5 routing payloads.

    The existing transpose-and-reduce workaround remains the fallback. Never
    restore the unreliable bool column reduction outside the PR's shape gate.
    """
    supported = (
        mask.device.type == "musa"
        and mask.device.index == torch.musa.current_device()
        and mask.dtype == torch.bool
        and mask.ndim == 2
        and mask.shape[1] == experts == 32
        and (24577 <= mask.shape[0] <= 28608 or 28673 <= mask.shape[0] <= 32704)
        and mask.is_contiguous()
        and mask.storage_offset() == 0
        and hidden.dtype == torch.bfloat16
        and hidden.ndim == 2
        and hidden.shape == (mask.shape[0], 2048)
        and hidden.is_contiguous()
        and hidden.device == mask.device
        and probabilities.device == mask.device
        and probabilities.dtype == torch.float32
        and probabilities.shape == (mask.shape[0], 8)
        and probabilities.is_contiguous()
    )
    if supported:
        COUNTERS["long_input_column_sum"] += 1
        return mask.to(torch.int64).sum(0, dtype=torch.long)
    COUNTERS["contiguous_row_sum"] += 1
    return mask.t().contiguous().sum(1, dtype=torch.long)
