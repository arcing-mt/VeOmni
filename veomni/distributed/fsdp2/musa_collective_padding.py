"""Default-off, per-model FSDP2 collective alignment for eight-rank MUSA groups."""

import os
from collections import Counter

import torch
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp._fully_shard import _fsdp_collectives as collectives

from ...utils.device import IS_MUSA_AVAILABLE


calls = Counter()
shapes = Counter()


def _eligible(output, source, group, gather):
    if (
        group.size() != 8
        or source.device.type != "musa"
        or output.device != source.device
        or output.dtype != source.dtype
        or source.dtype not in (torch.bfloat16, torch.float32)
        or not source.is_contiguous()
        or not output.is_contiguous()
        or source.ndim != 1
        or output.ndim != 1
    ):
        return None
    n = source.numel() if gather else output.numel()
    if n < 1_000_000 or n * source.element_size() % 16 == 0:
        return None
    if output.numel() != n * group.size() if gather else source.numel() != n * group.size():
        return None
    alignment = 16 // source.element_size()
    return n, ((n + alignment - 1) // alignment) * alignment


class MUSAAlignedAllGather(collectives.DefaultAllGather):
    @torch.no_grad()
    def __call__(self, output_tensor, input_tensor, group, async_op=False):
        size = _eligible(output_tensor, input_tensor, group, True)
        if size is None:
            return super().__call__(output_tensor, input_tensor, group, async_op)
        n, padded = size
        source = self.allocate((padded,), dtype=input_tensor.dtype, device=input_tensor.device)
        source[:n].copy_(input_tensor)
        source[n:].zero_()
        result = self.allocate((padded * group.size(),), dtype=output_tensor.dtype, device=output_tensor.device)
        work = super().__call__(result, source, group, async_op)
        if work is not None:
            # Establish the comm -> current-stream dependency before copying.
            # AllGatherResult's subsequent event covers this copy as well.
            work.wait()
        output_tensor.view(group.size(), n).copy_(result.view(group.size(), padded)[:, :n])
        calls["all_gather"] += 1
        shapes[f"AG:{n}->{padded}:{input_tensor.dtype}"] += 1
        return work


class MUSAAlignedReduceScatter(collectives.DefaultReduceScatter):
    @torch.no_grad()
    def __call__(self, output_tensor, input_tensor, group, op, async_op=False):
        size = _eligible(output_tensor, input_tensor, group, False)
        if size is None:
            return super().__call__(output_tensor, input_tensor, group, op, async_op)
        n, padded = size
        source = self.allocate((padded * group.size(),), dtype=input_tensor.dtype, device=input_tensor.device)
        rows = source.view(group.size(), padded)
        rows[:, :n].copy_(input_tensor.view(group.size(), n))
        rows[:, n:].zero_()
        result = self.allocate((padded,), dtype=output_tensor.dtype, device=output_tensor.device)
        work = super().__call__(result, source, group, op, async_op)
        if work is not None:
            work.wait()
        output_tensor.copy_(result[:n])
        calls["reduce_scatter"] += 1
        shapes[f"RS:{n}->{padded}:{input_tensor.dtype}"] += 1
        return work


def configure_musa_collective_padding(model):
    """Attach custom collectives to this model only, before its first forward."""
    if os.environ.get("VEOMNI_MUSA_FSDP_SHARD_PADDING", "0") != "1":
        return
    requested_overlap = os.environ.get("TORCH_MUSA_FSDP2_OVERLAP_LEVEL")
    if requested_overlap is None:
        requested_overlap = "4" if os.environ.get("TORCH_MUSA_FSDP2_DISABLE_OVERLAP", "1") == "0" else "0"
    if requested_overlap != "0" or os.environ.get("TORCH_MUSA_FSDP2_COMM_TYPE", "0") != "0":
        raise RuntimeError("MUSA shard padding requires FSDP2 OVERLAP_LEVEL=0 and COMM_TYPE=0")
    if IS_MUSA_AVAILABLE:
        from torch_musa.distributed._composable.fsdp.custom_overlap_patch import _FSDP2_OVERLAP_LEVEL

        if _FSDP2_OVERLAP_LEVEL.value != 0:
            raise RuntimeError("MUSA shard padding requires effective FSDP2 OVERLAP_LEVEL=0")
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_custom_all_gather(MUSAAlignedAllGather())
            module.set_custom_reduce_scatter(MUSAAlignedReduceScatter())
