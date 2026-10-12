"""Copy-only slot probabilities around the existing TE hidden permutation."""

import os
from collections import Counter

import torch
import triton
import triton.language as tl

from ....distributed.moe import deepep_ace as reference


COUNTERS = Counter()


@triton.jit
def _weights_from_slots(Indices, Probabilities, Weights, tokens, BLOCK: tl.constexpr, INDEX_OFFSET: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = offset // 32
    expert = offset % 32
    valid = token < tokens
    weight = tl.full((BLOCK,), 0, tl.float32)
    for slot in tl.static_range(8):
        routed_expert = tl.load(Indices + token * 8 + slot, valid, other=-1) - INDEX_OFFSET
        probability = tl.load(Probabilities + token * 8 + slot, valid, other=0)
        weight = tl.where(routed_expert == expert, probability, weight)
    tl.store(Weights + offset, weight, valid)


@triton.jit
def _gradient_to_slots(
    Indices, RowMap, GradProbabilities, GradSlots, Multipliers, tokens, BLOCK: tl.constexpr, INDEX_OFFSET: tl.constexpr
):
    slot = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = slot // 8
    expert = tl.load(Indices + slot, token < tokens, other=-1) - INDEX_OFFSET
    valid = (token < tokens) & (expert >= 0) & (expert < 32)
    row = tl.load(RowMap + expert * tokens + token, valid, other=-1)
    gradient = tl.load(GradProbabilities + row, valid & (row >= 0), other=0)
    tl.store(GradSlots + slot, gradient, token < tokens)
    tl.store(Multipliers + slot, (valid & (row >= 0)).to(tl.float32), token < tokens)


def slot_gradients(grad_probabilities, row_map, indices, *, index_offset=0, out=None):
    """Gather directly into top-k slots, optionally in a registered ACE window.

    ``index_offset=1`` accepts the checkpoint's stable ``recv_indices + 1``
    metadata without another device allocation. Return None for unsupported
    inputs so callers can retain their existing dense adjoint.
    """
    if (
        grad_probabilities is None
        or index_offset not in (0, 1)
        or indices.device.type != "musa"
        or indices.ndim != 2
        or indices.shape[1] != 8
        or indices.shape[0] == 0
        or indices.dtype != torch.int64
        or not indices.is_contiguous()
        or row_map.shape != (32, indices.shape[0])
        or row_map.dtype != torch.int64
        or not row_map.is_contiguous()
        or grad_probabilities.ndim != 1
        or grad_probabilities.numel() == 0
        or grad_probabilities.dtype != torch.float32
        or grad_probabilities.device != indices.device
        or row_map.device != indices.device
        or indices.device.index != torch.musa.current_device()
        or torch.is_grad_enabled()
    ):
        return None
    if out is not None and (
        out.shape != indices.shape
        or out.dtype != grad_probabilities.dtype
        or out.device != indices.device
        or not out.is_contiguous()
    ):
        return None
    contiguous = grad_probabilities.contiguous()
    grad_slots = torch.empty(indices.shape, device=indices.device, dtype=contiguous.dtype)
    multipliers = torch.empty_like(grad_slots)
    stream = torch.musa.current_stream()
    for value in (contiguous, row_map, indices):
        value.record_stream(stream)
    _gradient_to_slots[(triton.cdiv(indices.numel(), 1024),)](
        indices,
        row_map,
        contiguous,
        grad_slots,
        multipliers,
        indices.shape[0],
        BLOCK=1024,
        INDEX_OFFSET=index_offset,
        num_warps=4,
    )
    # Preserve the Torch FP32 multiplication used by the dense adjoint,
    # including its denormal behavior. ``out`` keeps ACE's window ownership.
    result = torch.mul(grad_slots, multipliers, out=out) if out is not None else grad_slots * multipliers
    COUNTERS["direct_slot_gradient"] += 1
    if out is not None:
        COUNTERS["registered_slot_gradient"] += 1
    return result


def _compact_forward(hidden, probabilities, indices, row_nt, tokens, width, out_tokens, index_offset):
    """Forward-only primitive for callers that own dispatch's adjoint."""
    _, native_permute, _, dtypes = reference._ace_te_impl()
    weights = torch.empty((tokens, width), device=probabilities.device, dtype=probabilities.dtype)
    stream = torch.musa.current_stream()
    indices.record_stream(stream)
    probabilities.record_stream(stream)
    _weights_from_slots[(triton.cdiv(tokens * width, 1024),)](
        indices, probabilities, weights, tokens, BLOCK=1024, INDEX_OFFSET=index_offset, num_warps=4
    )
    COUNTERS["fused_weights"] += 1
    empty = torch.empty(0, device="cpu")
    return native_permute(
        dtypes[hidden.dtype], hidden, row_nt, weights, tokens, width, out_tokens, hidden.shape[-1], empty, empty
    )


class _TECompactSlotProbabilities(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, probabilities, indices, row_nt, row_map, tokens, width, out_tokens, index_offset):
        ctx.save_for_backward(row_nt, row_map, indices.clone())
        ctx.tokens = tokens
        ctx.width = width
        ctx.index_offset = index_offset
        return _compact_forward(hidden, probabilities, indices, row_nt, tokens, width, out_tokens, index_offset)

    @staticmethod
    def backward(ctx, grad_hidden_output, grad_probabilities):
        _, _, native_unpermute, dtypes = reference._ace_te_impl()
        row_nt, row_map, indices = ctx.saved_tensors
        empty = torch.empty(0, device="cpu")
        grad_hidden = native_unpermute(
            dtypes[grad_hidden_output.dtype],
            grad_hidden_output.contiguous(),
            row_map,
            empty,
            empty,
            ctx.tokens,
            ctx.width,
            grad_hidden_output.shape[-1],
            empty,
            empty,
        )[0]
        grad_slots = None
        if grad_probabilities is not None:
            grad_slots = slot_gradients(grad_probabilities, row_map, indices, index_offset=ctx.index_offset)
            if grad_slots is None:
                dense = grad_probabilities[row_nt.clamp_min(0)] * (row_nt >= 0).to(grad_probabilities.dtype)
                grad_slots = torch.nn.functional.pad(dense, (1, 0)).gather(1, indices + 1 - ctx.index_offset)
                COUNTERS["higher_order_native_probability"] += 1
        return grad_hidden, grad_slots, None, None, None, None, None, None, None


def compact(hidden, probabilities, indices, row_nt, row_map, tokens, width, out_tokens, *, index_offset=0):
    """Fuse zero-fill/scatter/slice-copy into one dense TE weight producer.

    Routing validation remains with the caller. Native Qwen top-k routes are
    unique; replay routes must pass the existing duplicate/count guards first.
    """
    eligible = (
        all(type(value) is torch.Tensor for value in (hidden, probabilities, indices, row_nt, row_map))
        and tokens > 0
        and out_tokens > 0
        and width == 32
        and index_offset in (0, 1)
        and hidden.ndim == 2
        and hidden.dtype == torch.bfloat16
        and hidden.shape[-1] == 2048
        and hidden.is_contiguous()
        and probabilities.shape == (tokens, 8)
        and indices.shape == (tokens, 8)
        and probabilities.dtype == torch.float32
        and indices.dtype == torch.int64
        and probabilities.device.type == "musa"
        and probabilities.device == indices.device == hidden.device
        and hidden.device.index == torch.musa.current_device()
        and probabilities.is_contiguous()
        and indices.is_contiguous()
        and row_nt.dtype == row_map.dtype == torch.int64
        and row_map.shape == (width, tokens)
        and row_nt.shape == (tokens, width)
        and row_map.device == row_nt.device == hidden.device
        and row_nt.is_contiguous()
        and row_map.is_contiguous()
    )
    if eligible:
        if os.environ.get("VEOMNI_MUSA_ACE_COMPACT_STATE_REUSE", "0") == "1" and not torch.is_grad_enabled():
            COUNTERS["forward_only_compact"] += 1
            return _compact_forward(hidden, probabilities, indices, row_nt, tokens, width, out_tokens, index_offset)
        return _TECompactSlotProbabilities.apply(
            hidden, probabilities, indices, row_nt, row_map, tokens, width, out_tokens, index_offset
        )
    COUNTERS["native_weights"] += 1
    weights = torch.zeros((tokens, width + 1), dtype=torch.float32, device=hidden.device)
    weights.scatter_(1, (indices + 1 - index_offset).to(torch.int64), probabilities)
    weights = weights[:, 1:].contiguous()
    return reference._TECompactPermute.apply(hidden, weights, row_nt, row_map, tokens, width, out_tokens)
