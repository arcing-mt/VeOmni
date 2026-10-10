"""Process-only FSDP2 collective padding for the measured MUSA workload."""
from collections import Counter
import torch
from torch.distributed.fsdp._fully_shard import _fsdp_collectives as collectives

enabled = True
calls = Counter()
shapes = Counter()
_installed = False


def _eligible(output, source, group, gather):
    if (not enabled or group.size() != 8 or source.device.type != 'musa'
            or output.device != source.device or output.dtype != source.dtype
            or source.dtype not in (torch.bfloat16, torch.float32)
            or not source.is_contiguous() or not output.is_contiguous()
            or source.ndim != 1 or output.ndim != 1):
        return None
    n = source.numel() if gather else output.numel()
    if n < 1_000_000 or n * source.element_size() % 16 == 0:
        return None
    if (output.numel() != n * group.size() if gather else source.numel() != n * group.size()):
        return None
    alignment = 16 // source.element_size()
    return n, ((n + alignment - 1) // alignment) * alignment


def install():
    global _installed
    if _installed:
        return
    original_ag = collectives.DefaultAllGather.__call__
    original_rs = collectives.DefaultReduceScatter.__call__

    @torch.no_grad()
    def all_gather(self, output_tensor, input_tensor, group, async_op=False):
        size = _eligible(output_tensor, input_tensor, group, True)
        if size is None:
            return original_ag(self, output_tensor, input_tensor, group, async_op)
        n, padded = size
        source = self.allocate((padded,), dtype=input_tensor.dtype, device=input_tensor.device)
        source[:n].copy_(input_tensor)
        source[n:].zero_()
        result = self.allocate((padded * group.size(),), dtype=output_tensor.dtype, device=output_tensor.device)
        work = original_ag(self, result, source, group, async_op)
        if work is not None:
            # Establish the comm -> current-stream dependency before copying.
            # AllGatherResult's subsequent event covers this copy as well.
            work.wait()
        output_tensor.view(group.size(), n).copy_(result.view(group.size(), padded)[:, :n])
        calls['all_gather'] += 1
        shapes[f'AG:{n}->{padded}:{input_tensor.dtype}'] += 1
        return work

    @torch.no_grad()
    def reduce_scatter(self, output_tensor, input_tensor, group, op, async_op=False):
        size = _eligible(output_tensor, input_tensor, group, False)
        if size is None:
            return original_rs(self, output_tensor, input_tensor, group, op, async_op)
        n, padded = size
        source = self.allocate((padded * group.size(),), dtype=input_tensor.dtype, device=input_tensor.device)
        rows = source.view(group.size(), padded)
        rows[:, :n].copy_(input_tensor.view(group.size(), n))
        rows[:, n:].zero_()
        result = self.allocate((padded,), dtype=output_tensor.dtype, device=output_tensor.device)
        work = original_rs(self, result, source, group, op, async_op)
        if work is not None:
            work.wait()
        output_tensor.copy_(result[:n])
        calls['reduce_scatter'] += 1
        shapes[f'RS:{n}->{padded}:{input_tensor.dtype}'] += 1
        return work

    collectives.DefaultAllGather.__call__ = all_gather
    collectives.DefaultReduceScatter.__call__ = reduce_scatter
    _installed = True


def stats():
    return {'calls': dict(calls), 'shapes': dict(shapes),
            'note': '16-byte shard padding; original dtype/op; copies and waits included; only default FSDP2 comm.'}
