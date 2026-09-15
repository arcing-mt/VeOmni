"""DeepEP-ACE token communication for the MUSA MoE dispatcher.

The dispatcher is deliberately separate from the expert compute backend. The
normal ``alltoall`` path remains in :mod:`moe_layer`; this module is entered
only when ``OpsImplementationConfig.moe_dispatcher`` is ``"deepep_ace"``.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

import torch
import torch.distributed as dist

from ...ops.config.singleton import get_ops_config
from ..parallel_state import get_parallel_state
from .moe_layer import EPGroupGemm, EPMergedFc1GroupGemm


def _supported_ep_classes() -> tuple[type, ...]:
    """Return grouped-GEMM autograd classes available on the active backend."""
    classes: list[type] = [EPGroupGemm, EPMergedFc1GroupGemm]
    try:
        from ...ops.kernels.moe.musa_group_gemm import (
            MusaEPGroupGemm,
            MusaEPMergedFc1GroupGemm,
        )
    except (ImportError, OSError):
        return tuple(classes)
    return (*classes, MusaEPGroupGemm, MusaEPMergedFc1GroupGemm)


def _load_deepep() -> tuple[Any, Any, Any]:
    try:
        from deep_ep import Buffer
        # This DeepEP wheel re-exports EventHandle from ``deep_ep.utils`` but
        # keeps EventOverlap in the concrete ``event`` module.  Import both
        # from that module so the supported wheel layout is handled directly.
        from deep_ep.utils.event import EventHandle, EventOverlap
    except ImportError as exc:  # pragma: no cover - hardware image dependent.
        raise RuntimeError(
            "moe_dispatcher='deepep_ace' requires a DeepEP build with MUSA ACE support"
        ) from exc
    return Buffer, EventHandle, EventOverlap


def _current_stream_event():
    """Return DeepEP's event wrapper for the current MUSA stream."""
    _, EventHandle, EventOverlap = _load_deepep()
    return EventOverlap(EventHandle())


_BUFFER_CACHE: dict[tuple[int, int, int], Any] = {}


class _ACEState:
    """Per-MoE-invocation handle state; the buffer itself is process cached."""

    def __init__(self, group: dist.ProcessGroup, num_experts: int, top_k: int):
        self.group = group
        self.num_experts = num_experts
        self.top_k = top_k
        self.handle = None

    def _get_buffer(self, hidden_states: torch.Tensor):
        Buffer, _, _ = _load_deepep()
        hidden_bytes = hidden_states.size(1) * max(hidden_states.element_size(), 2)
        config = get_ops_config()
        num_sms = getattr(config, "moe_deepep_num_sms", 20) if config is not None else 20
        cache_key = (id(self.group), hidden_bytes, num_sms)
        if cache_key in _BUFFER_CACHE:
            return _BUFFER_CACHE[cache_key]

        Buffer.set_num_sms(num_sms)
        dispatch_config = Buffer.get_dispatch_config(self.group.size())
        combine_config = Buffer.get_combine_config(self.group.size())
        nvl_bytes = max(
            dispatch_config.get_nvl_buffer_size_hint(hidden_bytes, self.group.size()),
            combine_config.get_nvl_buffer_size_hint(hidden_bytes, self.group.size()),
        )
        rdma_bytes = max(
            dispatch_config.get_rdma_buffer_size_hint(hidden_bytes, self.group.size()),
            combine_config.get_rdma_buffer_size_hint(hidden_bytes, self.group.size()),
        )
        buffer_kwargs = {"use_ace": True, "num_ace_buffers": 1}
        # ``train_mode`` exists in the reference llm_pretrain_script wheel,
        # but not in the currently installed DeepEP wheel.  Pass it only when
        # the constructor advertises the parameter; ACE itself is selected by
        # ``use_ace`` in both layouts.
        if "train_mode" in inspect.signature(Buffer).parameters:
            buffer_kwargs["train_mode"] = True
        buffer = Buffer(self.group, nvl_bytes, rdma_bytes, **buffer_kwargs)
        _BUFFER_CACHE[cache_key] = buffer
        return buffer

    def dispatch(self, hidden_states, selected_experts, routing_weights):
        buffer = self._get_buffer(hidden_states)
        previous_event = _current_stream_event()
        (
            tokens_per_rank,
            tokens_per_rdma_rank,
            tokens_per_expert,
            token_in_rank,
            layout_event,
        ) = buffer.get_dispatch_layout(
            selected_experts,
            self.num_experts,
            previous_event=previous_event,
            async_finish=True,
            allocate_on_comm_stream=True,
        )
        (
            recv_hidden,
            recv_indices,
            recv_probs,
            _recv_counts,
            handle,
            dispatch_event,
        ) = buffer.dispatch(
            hidden_states.contiguous(),
            topk_idx=selected_experts,
            topk_weights=routing_weights.float(),
            num_tokens_per_rank=tokens_per_rank,
            num_tokens_per_rdma_rank=tokens_per_rdma_rank,
            is_token_in_rank=token_in_rank,
            num_tokens_per_expert=tokens_per_expert,
            previous_event=layout_event,
            async_finish=True,
            allocate_on_comm_stream=True,
        )
        dispatch_event.current_stream_wait()
        self.handle = handle
        return recv_hidden, recv_indices, recv_probs

    def combine(self, expert_outputs):
        if self.handle is None:
            raise RuntimeError("DeepEP-ACE combine called without a dispatch handle")
        combined, _, event = self._get_buffer(expert_outputs).combine(
            expert_outputs.contiguous(),
            self.handle,
            previous_event=_current_stream_event(),
            async_finish=True,
            allocate_on_comm_stream=True,
        )
        event.current_stream_wait()
        return combined

    def reverse_dispatch(self, grad_output):
        if self.handle is None:
            raise RuntimeError("DeepEP-ACE reverse dispatch called without a handle")
        grad_recv, _, _, _, _, event = self._get_buffer(grad_output).dispatch(
            grad_output.contiguous(),
            handle=self.handle,
            previous_event=_current_stream_event(),
            async_finish=True,
            allocate_on_comm_stream=True,
        )
        event.current_stream_wait()
        return grad_recv

    def reverse_combine(self, grad_recv_hidden, grad_recv_probs):
        if self.handle is None:
            raise RuntimeError("DeepEP-ACE reverse combine called without a handle")
        grad_hidden, grad_probs, event = self._get_buffer(grad_recv_hidden).combine(
            grad_recv_hidden.contiguous(),
            self.handle,
            topk_weights=grad_recv_probs.float() if grad_recv_probs is not None else None,
            previous_event=_current_stream_event(),
            async_finish=True,
            allocate_on_comm_stream=True,
        )
        event.current_stream_wait()
        return grad_hidden, grad_probs


class _ACEDispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, selected_experts, routing_weights, state):
        recv_hidden, recv_indices, recv_probs = state.dispatch(
            hidden_states, selected_experts, routing_weights
        )
        ctx.state = state
        return recv_hidden, recv_indices, recv_probs

    @staticmethod
    def backward(ctx, grad_recv_hidden, _grad_indices, grad_recv_probs):
        grad_hidden, grad_probs = ctx.state.reverse_combine(
            grad_recv_hidden, grad_recv_probs
        )
        return grad_hidden, None, grad_probs, None


class _ACECombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, expert_outputs, state):
        ctx.state = state
        return state.combine(expert_outputs)

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.state.reverse_dispatch(grad_output), None


def _compact_permute(recv_hidden, recv_indices, recv_probs, num_local_experts):
    if recv_indices is None or recv_probs is None:
        raise RuntimeError("DeepEP-ACE did not return routing metadata")
    flat_indices = recv_indices.reshape(-1)
    valid_slots = torch.nonzero(flat_indices >= 0, as_tuple=False).flatten()
    experts = flat_indices.index_select(0, valid_slots).to(torch.long)
    order = torch.argsort(experts, stable=True)
    slots = valid_slots.index_select(0, order)
    sorted_experts = experts.index_select(0, order)
    token_rows = torch.div(slots, recv_indices.shape[1], rounding_mode="floor")
    permuted = recv_hidden.index_select(0, token_rows)
    probs = recv_probs.reshape(-1).index_select(0, slots)
    counts = torch.bincount(sorted_experts, minlength=num_local_experts).to(torch.long)
    return permuted, probs, token_rows, counts


def _compact_unpermute(expert_outputs, probs, token_rows, recv_tokens):
    weighted = expert_outputs * probs.to(expert_outputs.dtype).unsqueeze(-1)
    restored = torch.zeros(
        (recv_tokens, expert_outputs.shape[-1]),
        dtype=expert_outputs.dtype,
        device=expert_outputs.device,
    )
    return restored.index_add(0, token_rows, weighted)


def dispatch_to_ep_class_deepep_ace(
    ep_class: Callable[..., torch.Tensor],
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    *ep_class_args: Any,
) -> torch.Tensor:
    """Run ACE dispatch/combine while retaining the selected compute backend."""
    state = get_parallel_state()
    if not state.ep_enabled or state.ep_group is None or state.ep_group.size() <= 1:
        raise RuntimeError("moe_dispatcher='deepep_ace' requires expert parallelism")
    if ep_class not in _supported_ep_classes():
        raise NotImplementedError(
            "moe_dispatcher='deepep_ace' currently supports the non-LoRA EP grouped-GEMM paths only"
        )

    invocation = _ACEState(state.ep_group, num_experts, selected_experts.shape[-1])
    recv_hidden, recv_indices, recv_probs = _ACEDispatch.apply(
        hidden_states,
        selected_experts,
        routing_weights.float(),
        invocation,
    )
    num_local_experts = num_experts // state.ep_group.size()
    permuted, probs, token_rows, counts = _compact_permute(
        recv_hidden, recv_indices, recv_probs, num_local_experts
    )
    cumsum = counts.cumsum(0)
    expert_outputs = ep_class.apply(permuted, cumsum, *ep_class_args)
    restored = _compact_unpermute(expert_outputs, probs, token_rows, recv_hidden.shape[0])
    return _ACECombine.apply(restored, invocation)
