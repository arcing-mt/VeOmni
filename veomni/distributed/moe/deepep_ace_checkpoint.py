"""Selective checkpoint caching for ACE dispatch and local compaction."""

import contextvars
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import torch

from ...ops.config.singleton import get_ops_config


_ACTIVE_MOE_KEY = contextvars.ContextVar("veomni_active_moe_key", default=None)
_IN_CHECKPOINT_RECOMPUTE = contextvars.ContextVar("veomni_in_checkpoint_recompute", default=False)
_CHECKPOINT_DISPATCH_CACHE = contextvars.ContextVar("veomni_checkpoint_dispatch_cache", default=None)


def recompute_cache_enabled() -> bool:
    config = get_ops_config()
    return (
        config is not None
        and getattr(config, "moe_dispatcher", "alltoall") == "deepep_ace"
        and getattr(config, "moe_deepep_recompute_cache", "none") == "compact"
    )


def validate_model_cache_scopes(model: torch.nn.Module) -> None:
    blocks = [module for module in model.modules() if hasattr(module, "_veomni_cache_payload")]
    if not blocks or any(not getattr(block.forward, "_veomni_deepep_cache_scope", False) for block in blocks):
        raise ValueError(
            "DeepEP compact caching requires installed Qwen3.5-MoE cache scopes; "
            "use the VeOmni modeling backend on MUSA."
        )


class _CheckpointCacheContext:
    def __init__(self, records, recompute):
        self.records = records
        self.recompute = recompute

    def __enter__(self):
        self._cache_token = _CHECKPOINT_DISPATCH_CACHE.set(self.records)
        self._recompute_token = _IN_CHECKPOINT_RECOMPUTE.set(self.recompute)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _IN_CHECKPOINT_RECOMPUTE.reset(self._recompute_token)
        _CHECKPOINT_DISPATCH_CACHE.reset(self._cache_token)
        # Includes checkpoint early-stop. Records belong to this invocation,
        # never a global stack indexed by module or training step.
        if self.recompute or exc_type is not None:
            self.records.clear()


def deepep_recompute_context_fn():
    if not recompute_cache_enabled():
        return nullcontext(), nullcontext()
    records = []
    return _CheckpointCacheContext(records, False), _CheckpointCacheContext(records, True)


@contextmanager
def moe_dispatch_context(key: int, cache_payload: bool = True):
    token = _ACTIVE_MOE_KEY.set(key)
    cache_token = _CHECKPOINT_DISPATCH_CACHE.set(None) if not cache_payload else None
    try:
        yield
    finally:
        if cache_token is not None:
            _CHECKPOINT_DISPATCH_CACHE.reset(cache_token)
        _ACTIVE_MOE_KEY.reset(token)


def has_cached_dispatch() -> bool:
    return _CHECKPOINT_DISPATCH_CACHE.get() is not None


def _dispatch_signature(hidden, selected, weights, num_experts):
    return (
        _ACTIVE_MOE_KEY.get(),
        num_experts,
        tuple(hidden.shape),
        hidden.dtype,
        hidden.device,
        tuple(selected.shape),
        selected.dtype,
        tuple(weights.shape),
        weights.dtype,
    )


@dataclass
class _CompactRecord:
    signature: tuple
    handle: object
    received_tokens: int
    counts: list
    compact: torch.Tensor
    probabilities: torch.Tensor
    plan: dict
    event: object


class _DispatchCompact(torch.autograd.Function):
    """Retain only the expert input, with identical saves on both passes.

    The original received activation is temporary. Only this Function owns
    dispatch's adjoint; the expert GEMM and combine retain their own backward.
    """

    @staticmethod
    def forward(ctx, hidden, selected, weights, state):
        from . import deepep_ace as ace

        records = _CHECKPOINT_DISPATCH_CACHE.get()
        if records is None:
            raise RuntimeError("Compact dispatch caching requires a non-reentrant checkpoint context")
        signature = _dispatch_signature(hidden, selected, weights, state.num_experts)
        if _IN_CHECKPOINT_RECOMPUTE.get():
            if not records:
                raise RuntimeError("Missing compact dispatch checkpoint record")
            record = records.pop(0)
            if record.signature != signature:
                raise RuntimeError("Compact dispatch checkpoint invocation mismatch")
            record.event.current_stream_wait()
            state.handle = record.handle
            state.recv_counts = record.counts
            state.token_num = hidden.shape[0]
            state.received_tokens = record.received_tokens
            compact, probabilities, plan = record.compact, record.probabilities, record.plan
            overlap = ace._ACTIVE_SHARED_EXPERT.get()
            if overlap is not None:
                with torch.enable_grad():
                    overlap.start()
            if plan["backend"] == "te":
                row_map, row_nt = ace._ace_te_impl()[0](plan["multi_hot"], state.received_tokens, plan["width"])
                rows = None
            else:
                rows, row_map, row_nt = plan["token_rows"], None, None
            counts, width = plan["counts"], plan["width"]
        else:
            received, indices, received_weights = state.dispatch(
                hidden, selected, weights, getattr(state, "previous_event", None)
            )
            state.wait_dispatch()
            state.received_tokens = received.shape[0]
            plan = {}
            compact, probabilities, rows, counts, row_map, width = ace._compact_permute(
                received,
                indices,
                received_weights,
                state.num_experts // state.group.size(),
                state.recv_counts,
                plan=plan,
            )
            row_nt = row_map.t().contiguous() if row_map is not None else None
            records.append(
                _CompactRecord(
                    signature,
                    state.handle,
                    state.received_tokens,
                    list(state.recv_counts),
                    compact.detach(),
                    probabilities.detach(),
                    plan,
                    ace._current_stream_event(),
                )
            )

        ctx.state = state
        ctx.backend = plan["backend"]
        ctx.received_tokens = state.received_tokens
        ctx.width = width
        ctx.top_k = selected.shape[-1]
        if ctx.backend == "te":
            ctx.save_for_backward(row_nt, row_map, plan["scatter_indices"])
            ctx.mark_non_differentiable(counts, row_map)
        else:
            ctx.save_for_backward(rows, plan["slots"])
            ctx.mark_non_differentiable(rows, counts)
        return compact, probabilities, rows, counts, row_map, width

    @staticmethod
    def backward(ctx, grad_compact, grad_probabilities, _rows, _counts, _row_map, _width):
        from . import deepep_ace as ace

        if ctx.backend == "te":
            row_nt, row_map, scatter = ctx.saved_tensors
            _, _, unpermute, dtype = ace._ace_te_impl()
            empty = torch.empty(0, device="cpu")
            grad_received = unpermute(
                dtype[grad_compact.dtype],
                grad_compact.contiguous(),
                row_map,
                empty,
                empty,
                ctx.received_tokens,
                ctx.width,
                grad_compact.shape[-1],
                empty,
                empty,
            )[0]
            dense = grad_probabilities[row_nt.clamp_min(0)] * (row_nt >= 0).to(grad_probabilities.dtype)
            grad_weights = dense.gather(1, (scatter - 1).clamp_min(0)).masked_fill(scatter == 0, 0)
        else:
            rows, slots = ctx.saved_tensors
            grad_received = grad_compact.new_zeros((ctx.received_tokens, grad_compact.shape[-1]))
            grad_received.index_add_(0, rows, grad_compact)
            grad_weights = grad_probabilities.new_zeros((ctx.received_tokens * ctx.top_k,))
            grad_weights.index_add_(0, slots, grad_probabilities)
            grad_weights = grad_weights.view(ctx.received_tokens, ctx.top_k)
        grad_hidden, grad_router = ctx.state.reverse_combine(grad_received, grad_weights)
        return grad_hidden, None, grad_router, None
