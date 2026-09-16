"""DeepEP-ACE token communication for the MUSA MoE dispatcher.

The dispatcher is deliberately separate from the expert compute backend. The
normal ``alltoall`` path remains in :mod:`moe_layer`; this module is entered
only when ``OpsImplementationConfig.moe_dispatcher`` is ``"deepep_ace"``.
"""

from __future__ import annotations

import contextvars
import inspect
from contextlib import contextmanager
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
        raise RuntimeError("moe_dispatcher='deepep_ace' requires a DeepEP build with MUSA ACE support") from exc
    return Buffer, EventHandle, EventOverlap


def _current_stream_event():
    """Return DeepEP's event wrapper for the current MUSA stream."""
    _, EventHandle, EventOverlap = _load_deepep()
    return EventOverlap(EventHandle())


_BUFFER_CACHE: dict[tuple[int, int, int, int, int, int, int], Any] = {}
_CAPACITY_CHECKED: set[tuple[int, int]] = set()
_ACTIVE_SHARED_EXPERT = contextvars.ContextVar("veomni_active_shared_expert", default=None)


def _prioritize_backward(tensor: torch.Tensor | None) -> None:
    """Match the reference MoE stream scheduler's backward ordering hint."""
    grad_fn = getattr(tensor, "grad_fn", None)
    set_sequence_nr = getattr(grad_fn, "_set_sequence_nr", None)
    if set_sequence_nr is not None:
        set_sequence_nr(torch.iinfo(torch.int).max)


class _SharedExpertOverlap:
    """Queue an independent shared expert on a side stream and join it later."""

    def __init__(self, shared_expert: torch.nn.Module, shared_gate: torch.nn.Module, hidden_states: torch.Tensor):
        self.shared_expert = shared_expert
        self.shared_gate = shared_gate
        self.hidden_states = hidden_states
        self.stream = None
        self.event = None
        self.output = None

    def start(self) -> None:
        if not hasattr(torch, "musa"):
            raise RuntimeError("shared expert overlap requires torch.musa")
        # Keep the shared stream at the device default priority.  DeepEP's
        # communication stream may be high priority; making the compute stream
        # higher priority would serialize the ACE payload behind all shared
        # expert kernels instead of allowing the two to make progress.
        self.stream = getattr(self.shared_expert, "_veomni_shared_expert_stream", None)
        if self.stream is None:
            self.stream = torch.musa.Stream(priority=-1)
            self.shared_expert._veomni_shared_expert_stream = self.stream
        producer_event = torch.musa.current_stream().record_event()
        with torch.musa.stream(self.stream):
            self.stream.wait_event(producer_event)
            shared = self.shared_expert(self.hidden_states)
            self.output = torch.sigmoid(self.shared_gate(self.hidden_states)) * shared
            _prioritize_backward(self.output)
            self.event = self.stream.record_event()

    def finish(self) -> torch.Tensor:
        if self.event is None or self.output is None:
            raise RuntimeError("shared expert overlap was not started")
        torch.musa.current_stream().wait_event(self.event)
        return self.output


@contextmanager
def shared_expert_overlap(shared_expert: torch.nn.Module, shared_gate: torch.nn.Module, hidden_states: torch.Tensor):
    """Enable one ACE dispatch call to overlap the independent shared expert."""
    state = _SharedExpertOverlap(shared_expert, shared_gate, hidden_states)
    token = _ACTIVE_SHARED_EXPERT.set(state)
    try:
        yield state
    finally:
        _ACTIVE_SHARED_EXPERT.reset(token)


class _ACEState:
    """Per-MoE-invocation handle state; the buffer itself is process cached."""

    def __init__(self, group: dist.ProcessGroup, num_experts: int, top_k: int):
        self.group = group
        self.num_experts = num_experts
        self.top_k = top_k
        self.token_num = None
        self.handle = None
        self.dispatch_event = None
        self.recv_counts = None

    def _get_buffer(self, hidden_states: torch.Tensor):
        Buffer, _, _ = _load_deepep()
        hidden_size = hidden_states.size(1)
        element_size = hidden_states.element_size()
        hidden_bytes = hidden_size * max(element_size, 2)
        config = get_ops_config()
        num_sms = getattr(config, "moe_deepep_num_sms", 20) if config is not None else 20
        # ACE allocates a fixed token workspace in its C++ runtime.  All EP
        # ranks must use the same capacity, so it is an explicit config value
        # rather than a per-rank auto-growth decision.
        configured_capacity = getattr(config, "moe_deepep_token_capacity", 8192)
        if self.group.size() > 8:
            raise RuntimeError("DeepEP-ACE supports at most eight EP ranks in one node")
        capacity_check_key = (id(self.group), int(configured_capacity))
        if capacity_check_key not in _CAPACITY_CHECKED:
            capacities = [None] * self.group.size()
            dist.all_gather_object(capacities, int(configured_capacity), group=self.group)
            if any(capacity != int(configured_capacity) for capacity in capacities):
                raise RuntimeError(f"all EP ranks must use the same moe_deepep_token_capacity: received {capacities}")
            _CAPACITY_CHECKED.add(capacity_check_key)
        requested_tokens = self.token_num or hidden_states.size(0)
        if requested_tokens > configured_capacity:
            raise RuntimeError(
                "DeepEP-ACE input exceeds moe_deepep_token_capacity: "
                f"tokens={requested_tokens}, capacity={configured_capacity}. "
                "Increase the explicit capacity so every EP rank constructs the same workspace."
            )
        requested_capacity = ((int(configured_capacity) + 1023) // 1024) * 1024
        cache_prefix = (id(self.group), hidden_bytes, hidden_size, element_size, num_sms, self.top_k)
        reusable = [
            (key[6], buffer)
            for key, buffer in _BUFFER_CACHE.items()
            if key[:6] == cache_prefix and key[6] >= requested_capacity
        ]
        if reusable:
            return min(reusable, key=lambda item: item[0])[1]

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
        buffer_parameters = inspect.signature(Buffer).parameters
        if "token_num" in buffer_parameters:
            buffer_kwargs.update(
                token_num=requested_capacity,
                hidden_size=hidden_states.size(1),
                num_topk=self.top_k,
            )
        elif requested_capacity != 8192:
            raise RuntimeError(
                "the installed DeepEP wheel cannot configure moe_deepep_token_capacity; "
                "use a wheel exposing Buffer(token_num=..., hidden_size=..., num_topk=...)"
            )
        # ``train_mode`` exists in the reference llm_pretrain_script wheel,
        # but not in the currently installed DeepEP wheel.  Pass it only when
        # the constructor advertises the parameter; ACE itself is selected by
        # ``use_ace`` in both layouts.
        if "train_mode" in inspect.signature(Buffer).parameters:
            buffer_kwargs["train_mode"] = True
        buffer = Buffer(self.group, nvl_bytes, rdma_bytes, **buffer_kwargs)
        _BUFFER_CACHE[(*cache_prefix, requested_capacity)] = buffer
        return buffer

    def dispatch(self, hidden_states, selected_experts, routing_weights, previous_event=None):
        self.token_num = hidden_states.size(0)
        buffer = self._get_buffer(hidden_states)
        if previous_event is None:
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
            recv_counts,
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
        self.handle = handle
        # DeepEP already returns the local expert split sizes as host metadata.
        # Preserve them instead of rebuilding the same counts with a device
        # bincount after the dispatch event is joined.
        self.recv_counts = recv_counts
        self.dispatch_event = dispatch_event
        overlap = _ACTIVE_SHARED_EXPERT.get()
        if overlap is not None:
            # Custom autograd Function.forward runs under no_grad. Re-enable
            # recording for the independent module so its parameter/input
            # gradients remain connected when its result is joined outside
            # the communication Function.
            with torch.enable_grad():
                overlap.start()
        return recv_hidden, recv_indices, recv_probs

    def wait_dispatch(self) -> None:
        if self.dispatch_event is None:
            raise RuntimeError("DeepEP-ACE dispatch event is missing")
        self.dispatch_event.current_stream_wait()
        self.dispatch_event = None

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
        previous_event = getattr(state, "previous_event", None)
        recv_hidden, recv_indices, recv_probs = state.dispatch(
            hidden_states, selected_experts, routing_weights, previous_event
        )
        ctx.state = state
        return recv_hidden, recv_indices, recv_probs

    @staticmethod
    def backward(ctx, grad_recv_hidden, _grad_indices, grad_recv_probs):
        grad_hidden, grad_probs = ctx.state.reverse_combine(grad_recv_hidden, grad_recv_probs)
        return grad_hidden, None, grad_probs, None


class _ACECombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, expert_outputs, state):
        ctx.state = state
        return state.combine(expert_outputs)

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.state.reverse_dispatch(grad_output), None


def _compact_permute(recv_hidden, recv_indices, recv_probs, num_local_experts, expert_counts=None):
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
    if expert_counts is None:
        counts = torch.bincount(sorted_experts, minlength=num_local_experts).to(torch.long)
    else:
        if len(expert_counts) != num_local_experts:
            raise RuntimeError(
                f"DeepEP returned {len(expert_counts)} expert counts for {num_local_experts} local experts"
            )
        counts = torch.as_tensor(expert_counts, device=recv_hidden.device, dtype=torch.long)
        if sum(expert_counts) != int(slots.numel()):
            raise RuntimeError(
                "DeepEP expert counts do not match received routing slots: "
                f"counts={int(counts.sum().item())}, slots={int(slots.numel())}"
            )
    return permuted, probs, token_rows, counts


def _compact_unpermute(expert_outputs, probs, token_rows, recv_tokens):
    weighted = expert_outputs * probs.to(expert_outputs.dtype).unsqueeze(-1)
    restored = torch.zeros(
        (recv_tokens, expert_outputs.shape[-1]),
        dtype=expert_outputs.dtype,
        device=expert_outputs.device,
    )
    # Keep the destination allocation and accumulate in place.  The previous
    # out-of-place index_add created a second full [recv_tokens, hidden] tensor.
    restored.index_add_(0, token_rows, weighted)
    return restored


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
    overlap = _ACTIVE_SHARED_EXPERT.get()
    if overlap is not None:
        invocation.previous_event = _current_stream_event()
    recv_hidden, recv_indices, recv_probs = _ACEDispatch.apply(
        hidden_states,
        selected_experts,
        routing_weights.float(),
        invocation,
    )
    if overlap is not None:
        # This is the autograd output of the communication dispatch itself.
        # Raising this node matches the reference dispatch-postprocess hint;
        # setting the sequence on a later index_select would only reorder the
        # local compact operation.
        _prioritize_backward(recv_hidden)
    # This is immediately after the host-side dispatch call returns.  It must
    # stay outside the custom autograd Function: its forward executes with
    # grad recording disabled, while the shared expert needs a normal graph for
    # parameter and input gradients.
    # The received payload is consumed by local compaction below.
    invocation.wait_dispatch()
    num_local_experts = num_experts // state.ep_group.size()
    permuted, probs, token_rows, counts = _compact_permute(
        recv_hidden,
        recv_indices,
        recv_probs,
        num_local_experts,
        invocation.recv_counts,
    )
    cumsum = counts.cumsum(0)
    expert_outputs = ep_class.apply(permuted, cumsum, *ep_class_args)
    restored = _compact_unpermute(expert_outputs, probs, token_rows, recv_hidden.shape[0])
    output = _ACECombine.apply(restored, invocation)
    if overlap is not None:
        overlap.finish()
        output = output + overlap.output
    return output
