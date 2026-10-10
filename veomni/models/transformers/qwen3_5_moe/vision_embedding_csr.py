"""Default-off kernel for the fixed vision interpolation embedding site.

Forward remains native F.embedding. Only large (N>3072) BF16 positional
embedding backwards use stable CSR metadata, downloaded on the first grid
shape. The target wheel must pass direct parity; the public native source at a
different commit is only a mechanism hypothesis, not proof of its algorithm.
"""

import weakref
from collections import Counter, OrderedDict

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor
from torch.nn.modules import module as module_hooks


_cache = weakref.WeakKeyDictionary()
counters = Counter()
layouts = Counter()
MAX_CACHED_GRIDS = 128


@triton.jit
def _embedding_dw(gradient, slots, offsets, output, D: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0)
    dim = tl.arange(0, BD)
    begin = tl.load(offsets + row)
    end = tl.load(offsets + row + 1)
    accumulated = tl.full((BD,), 0.0, tl.float32)
    # Replicate FP32 sequential partials of at most10 entries, followed by
    # sequential FP32 partial summation. No tree/atomic accumulation or FMA.
    for chunk in range(begin, end, 10):
        partial = tl.full((BD,), 0.0, tl.float32)
        for j in tl.static_range(0, 10):
            valid = chunk + j < end
            slot = tl.load(slots + chunk + j, mask=valid, other=0)
            value = tl.load(gradient + slot * D + dim, mask=valid & (dim < D), other=0.0).to(tl.float32)
            partial = partial + value
        accumulated = accumulated + partial
    tl.store(output + row * D + dim, accumulated, mask=dim < D)


class _EmbeddingCSR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, indices, slots, offsets, ready):
        ctx.save_for_backward(slots, offsets, indices)
        ctx.ready = ready
        ctx.rows, ctx.width = weight.shape
        return F.embedding(indices, weight)

    @staticmethod
    def backward(ctx, gradient):
        slots, offsets, indices = ctx.saved_tensors
        if torch.is_grad_enabled():
            # Preserve the native differentiable adjoint for higher-order callers.
            counters["higher_order_native_backward"] += 1
            dw = torch.ops.aten.embedding_dense_backward(gradient, indices, ctx.rows, -1, False)
            return dw, None, None, None, None
        gradient = gradient.contiguous().reshape(-1, ctx.width)
        stream = torch.musa.current_stream(gradient.device)
        stream.wait_event(ctx.ready)
        slots.record_stream(stream)
        offsets.record_stream(stream)
        dw = torch.empty((ctx.rows, ctx.width), dtype=gradient.dtype, device=gradient.device)
        _embedding_dw[(ctx.rows,)](
            gradient, slots, offsets, dw, ctx.width, triton.next_power_of_2(ctx.width), num_warps=8
        )
        return dw, None, None, None, None


def _supported(module, indices, grid_key):
    return (
        not torch.is_inference_mode_enabled()
        and type(module) is torch.nn.Embedding
        and getattr(module.forward, "__func__", None) is torch.nn.Embedding.forward
        and not isinstance(module, FSDPModule)
        and not isinstance(module.weight, DTensor)
        and not any(
            (
                module._forward_hooks,
                module._forward_pre_hooks,
                module._backward_hooks,
                module._backward_pre_hooks,
                module_hooks._global_forward_hooks,
                module_hooks._global_forward_pre_hooks,
                module_hooks._global_backward_hooks,
                module_hooks._global_backward_pre_hooks,
            )
        )
        and module.padding_idx is None
        and module.max_norm is None
        and not module.scale_grad_by_freq
        and not module.sparse
        and isinstance(grid_key, tuple)
        and len(grid_key) == 3
        and all(isinstance(v, int) and v > 0 for v in grid_key)
        and grid_key[2] == 48
        and indices.device.type == module.weight.device.type == "musa"
        and indices.device == module.weight.device
        and module.weight.dtype == torch.bfloat16
        and tuple(module.weight.shape) == (2304, 1152)
        and module.weight.is_contiguous()
        and indices.dtype == torch.int64
        and indices.is_contiguous()
        and tuple(indices.shape) == (4, grid_key[0] * grid_key[1])
        and indices.numel() > 3072
    )


def cached_embedding(module, indices, grid_key):
    if not _supported(module, indices, grid_key):
        counters["fallbacks"] += 1
        return module(indices)
    # This helper may only replace the fixed version of the original vision
    # interpolation call-site; grid_key is its h/w/num_grid_per_side, not an
    # arbitrary user description of indices. Index values are shape-invariant
    # at that site. Different h/w with the same product remain distinct.
    key = (*grid_key, indices.device)
    module_cache = _cache.setdefault(module, OrderedDict())
    metadata = module_cache.get(key)
    if metadata is None:
        cpu = indices.detach().cpu().reshape(-1)
        assert bool(((cpu >= 0) & (cpu < 2304)).all())
        slots_cpu = torch.argsort(cpu, stable=True)
        counts = torch.bincount(cpu, minlength=2304)
        offsets_cpu = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        slots = slots_cpu.to(indices.device)
        offsets = offsets_cpu.to(indices.device)
        ready = torch.musa.Event()
        ready.record(torch.musa.current_stream(indices.device))
        metadata = (slots, offsets, ready)
        module_cache[key] = metadata
        if len(module_cache) > MAX_CACHED_GRIDS:
            module_cache.popitem(last=False)
            counters["evictions"] += 1
        counters["cache_misses"] += 1
    else:
        module_cache.move_to_end(key)
        counters["cache_hits"] += 1
    counters["calls"] += 1
    layouts[str(grid_key)] += 1
    slots, offsets, ready = metadata
    stream = torch.musa.current_stream(indices.device)
    stream.wait_event(ready)
    slots.record_stream(stream)
    offsets.record_stream(stream)
    return _EmbeddingCSR.apply(module.weight, indices, slots, offsets, ready)


def stats():
    return {
        "counters": dict(counters),
        "layouts": dict(layouts),
        "cached_modules": len(_cache),
        "cached_grids": sum(map(len, _cache.values())),
        "cache_limit_per_module": MAX_CACHED_GRIDS,
        "scope": "Fixed original interpolation indices only; native F.embedding FW, large BF16 CSR BW; hooks/DTensor/child FSDP fall back; first-cache download cost included.",
    }
