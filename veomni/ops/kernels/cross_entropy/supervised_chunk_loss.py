"""PR15 supervised causal-token compaction over the existing ChunkLoss adjoint."""

from collections import Counter

import torch

from ....distributed.parallel_state import get_parallel_state
from .chunk_loss import _native_chunk_loss_function as original


shapes = Counter()
fallbacks = 0


def compact_chunk_loss(
    hidden_states,
    weights,
    labels,
    chunk_size=1024,
    vocab_size=None,
    num_items_in_batch=None,
    ignore_index=-100,
    shift_labels=None,
    **kwargs,
):
    global fallbacks
    if get_parallel_state().sp_enabled or hidden_states.ndim != 3 or labels.ndim != 2:
        fallbacks += 1
        return original(
            hidden_states,
            weights,
            labels,
            chunk_size,
            vocab_size,
            num_items_in_batch,
            ignore_index,
            shift_labels,
            **kwargs,
        )
    shifted = labels[..., 1:] if shift_labels is None else shift_labels
    hidden = hidden_states[..., :-1, :] if shift_labels is None else hidden_states
    shifted = shifted.reshape(-1)
    selected = torch.nonzero(shifted != ignore_index, as_tuple=False).flatten()
    total, valid = shifted.numel(), selected.numel()
    shapes[(total, valid)] += 1
    if valid == 0 or valid == total:
        fallbacks += 1
        if shift_labels is None:
            return original(
                hidden_states,
                weights,
                labels,
                chunk_size,
                vocab_size,
                num_items_in_batch,
                ignore_index,
                **kwargs,
            )
        return original(
            hidden_states,
            weights,
            shift_labels,
            chunk_size,
            vocab_size,
            num_items_in_batch,
            ignore_index,
            _labels_already_shifted=True,
            **kwargs,
        )
    compact_hidden = hidden.reshape(-1, hidden.shape[-1]).index_select(0, selected).unsqueeze(0)
    compact_labels = shifted.index_select(0, selected).unsqueeze(0)
    # Compaction destroys sequence adjacency: never apply a second causal shift.
    # Keep the disabled public path's existing shift_labels behavior unchanged.
    return original(
        compact_hidden,
        weights,
        compact_labels,
        chunk_size,
        vocab_size,
        num_items_in_batch,
        ignore_index,
        _labels_already_shifted=True,
        **kwargs,
    )


def stats():
    total = sum(n * count for (n, v), count in shapes.items())
    valid = sum(v * count for (n, v), count in shapes.items())
    return {
        "calls": sum(shapes.values()),
        "fallbacks": fallbacks,
        "total_shifted_tokens": total,
        "supervised_tokens": valid,
        "fraction": valid / total if total else None,
        "shapes": {f"{n},{v}": count for (n, v), count in shapes.items()},
    }
