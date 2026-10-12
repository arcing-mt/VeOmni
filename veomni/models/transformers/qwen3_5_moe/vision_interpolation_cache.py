"""Bounded cache of immutable metadata for the source-guarded vision helper."""

import weakref
from collections import Counter, OrderedDict

import torch


_cache = weakref.WeakKeyDictionary()
counters = Counter()
MAX_CACHED_GRIDS = 128


def cached_metadata(module, height, width, grid, dtype, build):
    """Reuse indices and coefficients, never trainable position-table values."""
    device = torch.device(module.device)
    if device.type != "musa" or not all(type(x) is int and x > 0 for x in (height, width, grid)):
        counters["fallbacks"] += 1
        return build(module, height, width, grid, dtype)
    key = (height, width, grid, dtype, device)
    entries = _cache.setdefault(module, OrderedDict())
    metadata = entries.get(key)
    if metadata is None:
        # A cache populated by inference must still be usable in later training.
        with torch.inference_mode(False), torch.no_grad():
            indices, coefficients = build(module, height, width, grid, dtype)
            ready = torch.musa.Event()
            ready.record(torch.musa.current_stream(device))
        metadata = (indices, coefficients, ready)
        entries[key] = metadata
        if len(entries) > MAX_CACHED_GRIDS:
            entries.popitem(last=False)
            counters["evictions"] += 1
        counters["misses"] += 1
    else:
        entries.move_to_end(key)
        counters["hits"] += 1
    indices, coefficients, ready = metadata
    stream = torch.musa.current_stream(device)
    stream.wait_event(ready)
    indices.record_stream(stream)
    coefficients.record_stream(stream)
    return indices, coefficients


def stats():
    return {
        "counters": dict(counters),
        "cached_modules": len(_cache),
        "cached_grids": sum(len(entries) for entries in _cache.values()),
    }
