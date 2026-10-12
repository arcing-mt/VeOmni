"""Bounded integer reductions that never materialize an int64 routing mask."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N"])
def _partial_counts(mask, partials, N, BT: tl.constexpr):
    block = tl.program_id(0)
    tokens = block * BT + tl.arange(0, BT)
    experts = tl.arange(0, 32)
    values = tl.load(mask + tokens[:, None] * 32 + experts[None, :], tokens[:, None] < N, 0).to(tl.int32)
    counts = tl.sum(values, axis=0)
    tl.store(partials + block * 32 + experts, counts)


@triton.jit(do_not_specialize=["BLOCKS"])
def _finish_counts(partials, output, BLOCKS, B: tl.constexpr):
    blocks = tl.arange(0, B)
    experts = tl.arange(0, 32)
    values = tl.load(partials + blocks[:, None] * 32 + experts[None, :], blocks[:, None] < BLOCKS, 0)
    tl.store(output + experts, tl.sum(values, axis=0).to(tl.int64))


def fused_column_counts(mask):
    """Use int32 partials only within the guarded 65536-token capacity."""
    if (
        type(mask) is not torch.Tensor
        or mask.device.type != "musa"
        or mask.dtype != torch.bool
        or mask.ndim != 2
        or mask.shape[1] != 32
        or not 0 < mask.shape[0] <= 65536
        or not mask.is_contiguous()
    ):
        return None
    blocks = triton.cdiv(mask.shape[0], 256)
    partials = torch.empty((blocks, 32), dtype=torch.int32, device=mask.device)
    output = torch.empty(32, dtype=torch.int64, device=mask.device)
    with torch.musa.device(mask.device):
        _partial_counts[(blocks,)](mask, partials, N=mask.shape[0], BT=256, num_warps=4)
        _finish_counts[(1,)](partials, output, BLOCKS=blocks, B=triton.next_power_of_2(blocks), num_warps=4)
    return output
