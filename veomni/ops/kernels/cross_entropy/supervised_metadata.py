"""CPU-derived causal selection metadata for the opt-in supervised loss."""

from collections import Counter

import torch


KEY = "_veomni_supervised_ce"
counters = Counter()


def prepare_metadata(batch):
    """Run after packing/masking, while labels still live on the CPU."""
    labels = batch.get("labels")
    if not isinstance(labels, torch.Tensor) or labels.device.type != "cpu" or labels.ndim != 2:
        return
    shifted = labels[..., 1:].reshape(-1)
    indices = torch.nonzero(shifted != -100, as_tuple=False).flatten()
    batch.setdefault("multimodal_metadata", {})[KEY] = {
        "label_shape": tuple(labels.shape),
        "indices": indices,
        "targets": shifted.index_select(0, indices),
    }


def selection(labels, hidden, shift_labels, ignore_index, metadata):
    """Validate metadata structure; missing or unsupported inputs stay native.

    Values are owned by the final CPU packing stage, just like the existing
    multimodal offsets. Callers that change labels afterwards must remove this
    metadata. Explicitly shifted or SP labels never consume this causal cache.
    """
    cached = metadata.get(KEY) if isinstance(metadata, dict) else None
    if isinstance(cached, dict) and shift_labels is None and ignore_index == -100:
        indices, targets = cached.get("indices"), cached.get("targets")
        if (
            tuple(cached.get("label_shape", ())) == tuple(labels.shape) == tuple(hidden.shape[:-1])
            and type(indices) is torch.Tensor
            and type(targets) is torch.Tensor
            and indices.ndim == targets.ndim == 1
            and indices.dtype == torch.int64
            and targets.dtype == labels.dtype
            and indices.device == targets.device == labels.device
            and indices.numel() == targets.numel()
            and indices.numel() <= labels.shape[0] * max(labels.shape[1] - 1, 0)
        ):
            counters["hits"] += 1
            return indices, targets
    counters["fallbacks"] += 1
    return None


def stats():
    return dict(counters)
