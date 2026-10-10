"""Default-off MUSA ACE compaction with native TE hidden permutation.

Routing checks, Long counts and slot probability copies form one matched
permute/unpermute path. Unsupported inputs return to the original ACE code.
"""

import os
import sys
from collections import Counter
from dataclasses import dataclass

import torch
import triton

from veomni.utils import logging


logger = logging.get_logger(__name__)
_ACE_TE_CACHE = []
last_fallback_reason = None
MERGE_CHECKS = False
COUNTERS = Counter()
_LOAD_FAILED = False


def _validate_native_api(tex):
    if not all(callable(getattr(tex, name, None)) for name in ("moe_permute_mask", "moe_unpermute_mask")):
        raise ImportError("TransformerEngine lacks the qualified native compaction API")


def _ace_te_impl():
    if _ACE_TE_CACHE:
        return _ACE_TE_CACHE[0]
    import torch_musa

    if torch.__version__ != "2.11.0.post2" or torch_musa.__version__ != "2.11.0.post2+395c00c":
        raise RuntimeError("TE compaction is qualified only on the frozen Torch/MUSA build")
    if "transformer_engine" in sys.modules:
        raise RuntimeError(
            "TE was already imported; its original Torch API state cannot be recovered by this adapter."
        )
    surfaces = (torch, torch.Tensor, torch.nn.Module, torch.cuda, torch.cuda.nvtx, torch.distributed)
    originals = [(surface, dict(vars(surface))) for surface in surfaces]
    try:
        from transformer_engine.pytorch import cpp_extensions as tex
        from transformer_engine.pytorch.constants import TE_DType
        from transformer_engine.pytorch.triton.permutation import _row_id_map_pass_1_kernel, _row_id_map_pass_2_kernel
    finally:
        # TE's MUSA compatibility layer rewrites global Torch functions. The
        # native kernels used here need none of those Python-level aliases.
        for surface, attributes in originals:
            current = vars(surface)
            for name, value in attributes.items():
                if current.get(name) is not value:
                    setattr(surface, name, value)

    _validate_native_api(tex)

    def make_map(routing, tokens, experts):
        rows = torch.empty((experts, tokens), device=routing.device, dtype=torch.int64)
        transposed = torch.empty((tokens, experts), device=routing.device, dtype=torch.int64)
        block = 256
        grid = (experts, triton.cdiv(tokens, block))
        workspace = torch.empty(grid, device=routing.device, dtype=torch.int64)
        _row_id_map_pass_1_kernel[grid](routing, rows, workspace, tokens, routing.stride(0), routing.stride(1), block)
        _row_id_map_pass_2_kernel[grid](
            rows, transposed, workspace, experts, tokens, triton.next_power_of_2(experts * grid[1]), block
        )
        return rows, transposed

    _ACE_TE_CACHE[:] = [(make_map, tex.moe_permute_mask, tex.moe_unpermute_mask, TE_DType)]
    return _ACE_TE_CACHE[0]


def _ace_te_supported(hidden, indices, probabilities, experts):
    """Conservatively admit only the qualified MUSA payload family."""
    return (
        all(type(value) is torch.Tensor for value in (hidden, indices, probabilities))
        and hidden.device.type == "musa"
        and hidden.ndim == 2
        and hidden.shape[1] == 2048
        and hidden.dtype == torch.bfloat16
        and hidden.is_contiguous()
        and experts == 32
        and indices.shape == probabilities.shape == (hidden.shape[0], 8)
        and indices.dtype == torch.int64
        and probabilities.dtype == torch.float32
        and indices.device == probabilities.device == hidden.device
        and indices.is_contiguous()
        and probabilities.is_contiguous()
    )


class _TECompactPermute(torch.autograd.Function):
    """Gather the received rows into the expert-major layout with TE's permute.

    One fused kernel replaces the two ``index_select`` calls (activations and
    routing weights) of the PyTorch path.  Its adjoint is TE's un-permute: the
    autograd-generated adjoint of ``index_select`` accumulates duplicate rows with
    BF16 atomics, which on this workload measured 4.0e-3 against an FP64 reference
    and was not run-to-run reproducible, whereas TE sums in FP32 in a fixed order
    (6.9e-7, bit-reproducible).
    """

    @staticmethod
    def forward(ctx, recv_hidden, recv_probs, row_nt, row_map, num_tokens, width, num_out):
        _, tex_permute, _, te_dtype = _ace_te_impl()
        ctx.save_for_backward(row_nt, row_map)
        ctx.num_tokens = num_tokens
        ctx.width = width
        empty = torch.empty(0, device="cpu")
        return tex_permute(
            te_dtype[recv_hidden.dtype],
            recv_hidden,
            row_nt,
            recv_probs,
            num_tokens,
            width,
            num_out,
            recv_hidden.shape[-1],
            empty,
            empty,
        )

    @staticmethod
    def backward(ctx, grad_output, grad_probs):
        _, _, tex_unpermute, te_dtype = _ace_te_impl()
        row_nt, row_map = ctx.saved_tensors
        empty = torch.empty(0, device="cpu")
        grad_hidden = tex_unpermute(
            te_dtype[grad_output.dtype],
            grad_output.contiguous(),
            row_map,
            empty,
            empty,
            ctx.num_tokens,
            ctx.width,
            grad_output.shape[-1],
            empty,
            empty,
        )[0]
        if grad_probs is not None:
            grad_probs = grad_probs[row_nt.clamp_min(0)] * (row_nt >= 0).to(grad_probs.dtype)
        return (grad_hidden, grad_probs, None, None, None, None, None)


class _TECompactUnpermute(torch.autograd.Function):
    """Sum the expert-major rows back onto their received rows with TE kernels.

    The forward is TE's un-permute: one block per output row, the accumulator held
    in FP32 registers, contributions visited in a fixed order with ``-1`` slots
    skipped, so it never reads a padded row and is bit-reproducible.  The backward
    is TE's permute, the exact adjoint: measured against the ``index_select`` it
    replaces it is 31% faster (16-byte vector moves instead of int64-indexed row
    copies) and bit-identical to it.
    """

    @staticmethod
    def forward(ctx, weighted, row_id_map, width, num_tokens):
        _, _, tex_unpermute, te_dtype = _ace_te_impl()
        ctx.save_for_backward(row_id_map.t().contiguous())
        ctx.width = width
        ctx.num_tokens = num_tokens
        ctx.num_out = weighted.shape[0]
        empty = torch.empty(0, device="cpu")
        return tex_unpermute(
            te_dtype[weighted.dtype],
            weighted,
            row_id_map,
            empty,
            empty,
            num_tokens,
            width,
            weighted.shape[-1],
            empty,
            empty,
        )[0]

    @staticmethod
    def backward(ctx, grad_output):
        _, tex_permute, _, te_dtype = _ace_te_impl()
        (row_nt,) = ctx.saved_tensors
        empty = torch.empty(0, device="cpu")
        grad_weighted = tex_permute(
            te_dtype[grad_output.dtype],
            grad_output.contiguous(),
            row_nt,
            torch.zeros_like(row_nt, dtype=torch.float32),
            ctx.num_tokens,
            ctx.width,
            ctx.num_out,
            grad_output.shape[-1],
            empty,
            empty,
        )[0]
        return (grad_weighted, None, None, None)


def _te_compact_permute(recv_hidden, recv_indices, recv_probs, num_local_experts, expert_counts):
    """TE compaction: TE row ids plus one fused gather.

    Returns ``(permuted, probs, None, counts, row_id_map, num_local_experts)``, or
    ``None`` when this routing cannot take the TE path.
    """
    global last_fallback_reason
    last_fallback_reason = None
    make_row_id_map = _ace_te_impl()[0]
    num_tokens = recv_hidden.shape[0]
    multi_hot = torch.zeros((num_tokens, num_local_experts + 1), dtype=torch.bool, device=recv_hidden.device)
    multi_hot.scatter_(1, (recv_indices + 1).to(torch.int64), True)
    multi_hot = multi_hot[:, 1:].contiguous()
    if os.environ.get("VEOMNI_MUSA_ACE_TE_LONG_COUNTS", "0") == "1":
        from .musa_te_counts import column_counts

        counts = column_counts(multi_hot, recv_hidden, recv_probs, num_local_experts)
    else:
        counts = multi_hot.sum(0, dtype=torch.long)
    if MERGE_CHECKS:
        matches = (
            (counts == torch.as_tensor(expert_counts, device=recv_hidden.device, dtype=torch.long)).all()
            if expert_counts is not None
            else counts.new_ones(())
        )
        num_out, slot_total, counts_match = torch.stack(
            [counts.sum(), (recv_indices >= 0).sum(), matches.to(torch.long)]
        ).tolist()
    else:
        num_out, slot_total = torch.stack([counts.sum(), (recv_indices >= 0).sum()]).tolist()
        counts_match = None
    if num_out != slot_total:
        last_fallback_reason = "duplicate_expert_slots"
        return None
    if expert_counts is not None and (
        not counts_match
        if MERGE_CHECKS
        else not torch.equal(counts, torch.as_tensor(expert_counts, device=recv_hidden.device, dtype=torch.long))
    ):
        last_fallback_reason = "expert_counts_mismatch"
        return None
    row_id_map, row_nt = make_row_id_map(multi_hot, num_tokens, num_local_experts)
    if os.environ.get("VEOMNI_MUSA_ACE_TE_SLOT_PROBABILITY", "0") == "1":
        from .musa_te_slot_probability import compact

        permuted, permuted_probs = compact(
            recv_hidden, recv_probs, recv_indices, row_nt, row_id_map, num_tokens, num_local_experts, num_out
        )
    else:
        weights = torch.zeros((num_tokens, num_local_experts + 1), dtype=torch.float32, device=recv_hidden.device)
        weights.scatter_(1, (recv_indices + 1).to(torch.int64), recv_probs)
        weights = weights[:, 1:].contiguous()
        permuted, permuted_probs = _TECompactPermute.apply(
            recv_hidden, weights, row_nt, row_id_map, num_tokens, num_local_experts, num_out
        )
    return (permuted, permuted_probs, None, counts, row_id_map, num_local_experts)


@dataclass(frozen=True)
class TECompactionMetadata:
    row_map: torch.Tensor
    width: int


def try_compact(hidden, indices, probabilities, experts, expert_counts=None):
    """Return a matched TE payload or None; never import TE for disabled/CPU inputs."""
    global _LOAD_FAILED
    if os.environ.get("VEOMNI_MUSA_ACE_TE", "0") != "1" or hidden.device.type != "musa":
        return None
    if (
        hidden.shape[0] == 0
        or (expert_counts is not None and sum(expert_counts) == 0)
        or hidden.device.index != torch.musa.current_device()
    ):
        COUNTERS["empty_or_device_fallback"] += 1
        return None
    if not _ace_te_supported(hidden, indices, probabilities, experts):
        COUNTERS["layout_fallback"] += 1
        return None
    if expert_counts is not None and len(expert_counts) != experts:
        return None
    if _LOAD_FAILED:
        return None
    try:
        impl = _ace_te_impl()
    except (ImportError, OSError, RuntimeError) as exc:
        _LOAD_FAILED = True
        COUNTERS["dependency_fallback"] += 1
        logger.warning_rank0(f"MUSA ACE TE compaction unavailable; retaining native compaction: {exc}")
        return None
    if impl is None:
        COUNTERS["layout_fallback"] += 1
        return None
    result = _te_compact_permute(hidden, indices, probabilities, experts, expert_counts)
    if result is None:
        COUNTERS[last_fallback_reason or "routing_fallback"] += 1
        return None
    y, p, _, counts, row_map, width = result
    if y.shape[0] == 0:
        COUNTERS["empty_output_fallback"] += 1
        return None
    COUNTERS["te_compaction"] += 1
    return y, p, TECompactionMetadata(row_map, width), counts


def unpermute(expert_outputs, probs, metadata, recv_tokens):
    weighted = expert_outputs * probs.to(expert_outputs.dtype).unsqueeze(-1)
    return _TECompactUnpermute.apply(weighted, metadata.row_map, metadata.width, recv_tokens)
