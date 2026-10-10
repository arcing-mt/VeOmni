"""Copy-only slot probabilities around unchanged TE hidden permutation."""
from collections import Counter

import torch
import triton
import triton.language as tl
import te_compact_prototype as reference

COUNTERS = Counter()


@triton.jit
def _weights_from_slots(Indices, Probabilities, Weights, tokens, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = offset // 32
    expert = offset % 32
    valid = token < tokens
    weight = tl.full((BLOCK,), 0, tl.float32)
    for slot in tl.static_range(8):
        routed_expert = tl.load(Indices + token * 8 + slot, valid, other=-1)
        probability = tl.load(Probabilities + token * 8 + slot, valid, other=0)
        weight = tl.where(routed_expert == expert, probability, weight)
    tl.store(Weights + offset, weight, valid)


@triton.jit
def _gradient_to_slots(Indices, RowMap, GradProbabilities, GradSlots, Multipliers, tokens, BLOCK: tl.constexpr):
    slot = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = slot // 8
    expert = tl.load(Indices + slot, token < tokens, other=-1)
    valid = (token < tokens) & (expert >= 0) & (expert < 32)
    row = tl.load(RowMap + expert * tokens + token, valid, other=-1)
    gradient = tl.load(GradProbabilities + row, valid & (row >= 0), other=0)
    tl.store(GradSlots + slot, gradient, token < tokens)
    tl.store(Multipliers + slot, (valid & (row >= 0)).to(tl.float32), token < tokens)


class _TECompactSlotProbabilities(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, probabilities, indices, row_nt, row_map, tokens, width, out_tokens):
        _, native_permute, _, dtypes = reference._ace_te_impl()
        # The original scatter saves a newly computed indices+1 tensor.
        # Snapshot routing here as well: DeepEP may reuse its input buffer.
        ctx.save_for_backward(row_nt, row_map, indices.clone())
        ctx.tokens = tokens
        ctx.width = width
        weights = torch.empty((tokens, width), device=probabilities.device, dtype=probabilities.dtype)
        stream = torch.musa.current_stream()
        indices.record_stream(stream)
        probabilities.record_stream(stream)
        _weights_from_slots[(triton.cdiv(tokens * width, 1024),)](indices, probabilities, weights, tokens, BLOCK=1024, num_warps=4)
        COUNTERS["fused_weights"] += 1
        empty = torch.empty(0, device="cpu")
        return native_permute(dtypes[hidden.dtype], hidden, row_nt, weights, tokens, width, out_tokens, hidden.shape[-1], empty, empty)

    @staticmethod
    def backward(ctx, grad_hidden_output, grad_probabilities):
        _, _, native_unpermute, dtypes = reference._ace_te_impl()
        row_nt, row_map, indices = ctx.saved_tensors
        empty = torch.empty(0, device="cpu")
        # Identical native fixed-order FP32 hidden-gradient accumulation.
        grad_hidden = native_unpermute(dtypes[grad_hidden_output.dtype], grad_hidden_output.contiguous(), row_map, empty, empty, ctx.tokens, ctx.width, grad_hidden_output.shape[-1], empty, empty)[0]
        grad_slots = None
        if grad_probabilities is not None:
            if torch.is_grad_enabled():
                # Retain the original probability higher-order graph. This
                # does not expand the native TE hidden path's capabilities.
                dense = grad_probabilities[row_nt.clamp_min(0)] * (row_nt >= 0).to(grad_probabilities.dtype)
                grad_slots = torch.nn.functional.pad(dense, (1, 0)).gather(1, indices + 1)
                COUNTERS["higher_order_native_probability"] += 1
            else:
                contiguous = grad_probabilities.contiguous()
                grad_slots = torch.empty(indices.shape, device=contiguous.device, dtype=contiguous.dtype)
                multipliers = torch.empty_like(grad_slots)
                stream = torch.musa.current_stream()
                contiguous.record_stream(stream)
                row_map.record_stream(stream)
                indices.record_stream(stream)
                _gradient_to_slots[(triton.cdiv(ctx.tokens * 8, 1024),)](indices, row_map, contiguous, grad_slots, multipliers, ctx.tokens, BLOCK=1024, num_warps=4)
                # Keep the same Torch/MUSA FP32 multiplication primitive as
                # the original dense gp*mask. A plain copy (or compiler-
                # folded Triton *1) can differ for signaling NaNs/denormals.
                grad_slots = grad_slots * multipliers
                COUNTERS["direct_slot_gradient"] += 1
        return grad_hidden, grad_slots, None, None, None, None, None, None


def compact(hidden, probabilities, indices, row_nt, row_map, tokens, width, out_tokens):
    # Called only AFTER the original multi-hot scatter, counts/duplicate
    # checks and row map construction. Invalid indices and duplicates keep
    # their original validation/fallback, before any new kernel launch.
    eligible = (all(type(value) is torch.Tensor for value in (hidden, probabilities, indices, row_nt, row_map))
                and tokens > 0 and out_tokens > 0 and width == 32 and hidden.ndim == 2
                and hidden.dtype == torch.bfloat16 and hidden.shape[-1] == 2048 and hidden.is_contiguous()
                and probabilities.shape == (tokens, 8) and indices.shape == (tokens, 8)
                and probabilities.dtype == torch.float32 and indices.dtype == torch.int64
                and probabilities.device.type == "musa" and probabilities.device == indices.device == hidden.device
                and hidden.device.index == torch.musa.current_device()
                and probabilities.is_contiguous() and indices.is_contiguous()
                and row_nt.dtype == row_map.dtype == torch.int64
                and row_map.shape == (width, tokens) and row_nt.shape == (tokens, width)
                and row_map.device == row_nt.device == hidden.device
                and row_map.is_contiguous() and row_nt.is_contiguous())
    if eligible:
        return _TECompactSlotProbabilities.apply(hidden, probabilities, indices, row_nt, row_map, tokens, width, out_tokens)
    COUNTERS["native_weights"] += 1
    weights = torch.zeros((tokens, width + 1), dtype=torch.float32, device=hidden.device)
    weights.scatter_(1, (indices + 1).to(torch.int64), probabilities)
    weights = weights[:, 1:].contiguous()
    return reference._TECompactPermute.apply(hidden, weights, row_nt, row_map, tokens, width, out_tokens)


def stats():
    return dict(COUNTERS)
