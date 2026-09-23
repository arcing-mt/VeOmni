"""DeepEP-ACE token communication for the MUSA MoE dispatcher.

The dispatcher is deliberately separate from the expert compute backend. The
normal ``alltoall`` path remains in :mod:`moe_layer`; this module is entered
only when ``OpsImplementationConfig.moe_dispatcher`` is ``"deepep_ace"``.
"""

from __future__ import annotations

import contextvars
import inspect
import os
from contextlib import contextmanager
from typing import Any, Callable

import torch
import torch.distributed as dist

from ...ops.config.singleton import get_ops_config
from ...utils import logging
from ..parallel_state import get_parallel_state
from .moe_layer import EPGroupGemm, EPMergedFc1GroupGemm


logger = logging.get_logger(__name__)


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


class _SharedExpertOverlap:
    """Run the independent shared expert inside the ACE dispatch window.

    :meth:`_ACEState.dispatch` submits the ACE dispatch and then calls
    :meth:`start`, so the shared expert is issued on the compute stream while
    the dispatch payload is still in flight; ``wait_dispatch`` only joins the
    payload afterwards.  That is the overlap ``moe_shared_expert_overlap``
    promises, and it is worth the shared expert's own forward time.

    The shared expert deliberately does **not** get a stream of its own.  An
    earlier revision queued it on a dedicated MUSA stream and joined it with an
    event.  Measured on Qwen3.5-35B-A3B (8x MTT S5000, EP8, FSDP2) that cost
    ~0.7 s/step -- more than the shared expert itself -- because every MoE layer
    gained a cross-stream ``record_event``/``wait_event`` pair plus a stream
    switch for each of its kernels, the autograd graph was split across two
    streams, and the compute stream drained while the side stream was still
    filling, so the GPU starved between layers.  On the compute stream the
    shared expert still hides under the dispatch and the regression disappears.
    """

    def __init__(self, shared_expert: torch.nn.Module, shared_gate: torch.nn.Module, hidden_states: torch.Tensor):
        self.shared_expert = shared_expert
        self.shared_gate = shared_gate
        self.hidden_states = hidden_states
        self.output = None
        self.stream = None

    def start(self) -> None:
        """Compute the shared expert on the current stream.

        Must be called after the ACE dispatch has been submitted and before its
        payload is awaited, which is what puts the shared expert inside the
        dispatch window (see :meth:`_ACEState.dispatch`).
        """
        if not hasattr(torch, "musa"):
            raise RuntimeError("shared expert overlap requires torch.musa")
        if self.output is not None:
            raise RuntimeError("shared expert overlap was already started")
        self.stream = torch.musa.current_stream()
        self.output = torch.sigmoid(self.shared_gate(self.hidden_states)) * self.shared_expert(self.hidden_states)

    def finish(self) -> torch.Tensor:
        """Return the shared expert result.

        The tensor is only ordered on the stream that ran :meth:`start`, and
        nothing else re-establishes that ordering now that there is no event to
        wait on, so verify it rather than relying on the caller.
        """
        if self.output is None:
            raise RuntimeError("shared expert overlap was not started")
        if torch.musa.current_stream() != self.stream:
            raise RuntimeError("shared expert overlap finished on a different stream than it was started on")
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
        self._buffer = None

    def _get_buffer(self, hidden_states: torch.Tensor):
        # One state serves one MoE layer invocation, which resolves the same
        # buffer for dispatch, combine and both backward collectives.  Resolve
        # it once: the lookup costs ~14 us and would otherwise repeat four times
        # per layer (~2 ms/step at Qwen3.5-35B-A3B's 40 forward invocations).
        if self._buffer is not None:
            return self._buffer
        self._buffer = self._resolve_buffer(hidden_states)
        return self._buffer

    def _resolve_buffer(self, hidden_states: torch.Tensor):
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


_ACE_TE_CACHE: list = []
_ACE_TE_FELL_BACK: list = []


def _ace_te_impl():
    """Return ``(make_row_id_map, moe_permute_mask, moe_unpermute_mask, TE_DType)``, or ``None``.

    The ACE compaction runs either entirely on TransformerEngine kernels or
    entirely on the PyTorch fallback.  The two are driven by different maps -- TE
    by ``[num_local_experts, recv_tokens]`` expert-major ids, the fallback by the
    ``[topk, recv_tokens]`` slot order -- so a half-swapped pair would read the
    wrong rows.  ``VEOMNI_ACE_TE=0`` forces the fallback.
    """
    if _ACE_TE_CACHE:
        return _ACE_TE_CACHE[0]
    impl = None
    if os.environ.get("VEOMNI_ACE_TE", "1").strip().lower() not in {"0", "false", "no", "off"}:
        try:
            from transformer_engine.pytorch import cpp_extensions as tex
            from transformer_engine.pytorch.constants import TE_DType
            from transformer_engine.pytorch.triton.permutation import make_row_id_map

            if hasattr(tex, "moe_permute_mask") and hasattr(tex, "moe_unpermute_mask"):
                impl = (make_row_id_map, tex.moe_permute_mask, tex.moe_unpermute_mask, TE_DType)
        except (ImportError, OSError):
            impl = None
    if impl is not None:
        logger.info_rank0("ACE compaction: using TransformerEngine kernels")
    _ACE_TE_CACHE.append(impl)
    return impl


def _ace_te_supported(recv_hidden, recv_probs, num_local_experts):
    """Whether the whole TE path can run on this payload."""
    if _ace_te_impl() is None:
        return False
    # MUSA reports through ``is_musa``; ``is_cuda`` only becomes true once the
    # TransformerEngine compatibility shim has loaded, so test both explicitly.
    if not (recv_hidden.is_cuda or getattr(recv_hidden, "is_musa", False)):
        return False
    # The MUSA mask kernels are 16-bit only, move 16-byte vectors, index their
    # map as int64, and read the payload as if it were packed.
    if recv_hidden.dtype not in (torch.bfloat16, torch.float16):
        return False
    if not recv_hidden.is_contiguous():
        return False
    if recv_hidden.shape[-1] % (16 // recv_hidden.element_size()):
        return False
    # The permute kernel takes the routing weights as FP32 and only has a fused
    # path when its map width -- the local expert count here -- divides by four.
    return recv_probs.dtype == torch.float32 and num_local_experts % 4 == 0


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
            # Every map entry owns exactly one permuted row, so the adjoint of the
            # weight gather is a plain gather with no accumulation.
            grad_probs = grad_probs[row_nt.clamp_min(0)] * (row_nt >= 0).to(grad_probs.dtype)
        return grad_hidden, grad_probs, None, None, None, None, None


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
        return grad_weighted, None, None, None


def _te_compact_permute(recv_hidden, recv_indices, recv_probs, num_local_experts, expert_counts):
    """TE compaction: TE row ids plus one fused gather.

    Returns ``(permuted, probs, None, counts, row_id_map, num_local_experts)``, or
    ``None`` when this routing cannot take the TE path.
    """
    make_row_id_map = _ace_te_impl()[0]
    num_tokens = recv_hidden.shape[0]
    # A multi-hot map holds each token's *set* of experts.  Padding slots go to an
    # extra column that is sliced away, so no two writes race for one cell.
    multi_hot = torch.zeros((num_tokens, num_local_experts + 1), dtype=torch.bool, device=recv_hidden.device)
    multi_hot.scatter_(1, (recv_indices + 1).to(torch.int64), True)
    multi_hot = multi_hot[:, 1:].contiguous()
    counts = multi_hot.sum(0, dtype=torch.long)
    # One device->host read yields both the number of permuted rows and whether a
    # slot was collapsed (the count the fallback's ``nonzero`` needs anyway).
    num_out, slot_total = torch.stack([counts.sum(), (recv_indices >= 0).sum()]).tolist()
    if num_out != slot_total:
        return None
    if expert_counts is not None and not torch.equal(
        counts, torch.as_tensor(expert_counts, device=recv_hidden.device, dtype=torch.long)
    ):
        return None
    row_id_map, row_nt = make_row_id_map(multi_hot, num_tokens, num_local_experts)
    # Routing weights in the same expert-major layout the permute kernel reads.
    weights = torch.zeros((num_tokens, num_local_experts + 1), dtype=torch.float32, device=recv_hidden.device)
    weights.scatter_(1, (recv_indices + 1).to(torch.int64), recv_probs)
    weights = weights[:, 1:].contiguous()
    permuted, permuted_probs = _TECompactPermute.apply(
        recv_hidden,
        weights,
        row_nt,
        row_id_map,
        num_tokens,
        num_local_experts,
        num_out,
    )
    return permuted, permuted_probs, None, counts, row_id_map, num_local_experts


def _torch_compact_permute(recv_hidden, recv_indices, recv_probs, num_local_experts, expert_counts):
    """Fallback compaction: ``nonzero`` + stable ``argsort`` + ``index_select``.

    Returns ``(permuted, probs, token_rows, counts, row_map, topk)`` with the
    ``[topk, recv_tokens]`` slot map the fallback un-permute consumes.
    """
    flat_indices = recv_indices.reshape(-1)
    valid_slots = torch.nonzero(flat_indices >= 0, as_tuple=False).flatten()
    experts = flat_indices.index_select(0, valid_slots).to(torch.long)
    order = torch.argsort(experts, stable=True)
    slots = valid_slots.index_select(0, order)
    sorted_experts = experts.index_select(0, order)
    topk = recv_indices.shape[1]
    token_rows = torch.div(slots, topk, rounding_mode="floor")
    num_out = slots.numel()
    if expert_counts is None:
        counts = torch.bincount(sorted_experts, minlength=num_local_experts).to(torch.long)
    else:
        counts = torch.as_tensor(expert_counts, device=recv_hidden.device, dtype=torch.long)
        if sum(expert_counts) != num_out:
            raise RuntimeError(
                "DeepEP expert counts do not match received routing slots: "
                f"counts={sum(expert_counts)}, slots={num_out}"
            )
    permuted = recv_hidden.index_select(0, token_rows)
    probs = recv_probs.reshape(-1).index_select(0, slots)
    # ``token_rows`` is this layout's map: the un-permute scatters by it and
    # ``width`` is the top-k slot count.  The expert-major map is not needed.
    return permuted, probs, token_rows, counts, None, topk


def _compact_permute(recv_hidden, recv_indices, recv_probs, num_local_experts, expert_counts=None):
    """Move the received rows into the expert-major layout the grouped GEMM wants.

    Returns ``(permuted, probs, token_rows, counts, row_map, width)``.  ``width``
    is the width of ``row_map``; ``token_rows`` is ``None`` on the TE path, where
    the un-permute is driven by ``row_map`` alone.
    """
    if recv_indices is None or recv_probs is None:
        raise RuntimeError("DeepEP-ACE did not return routing metadata")
    if expert_counts is not None and len(expert_counts) != num_local_experts:
        raise RuntimeError(f"DeepEP returned {len(expert_counts)} expert counts for {num_local_experts} local experts")
    if _ace_te_supported(recv_hidden, recv_probs, num_local_experts):
        built = _te_compact_permute(recv_hidden, recv_indices, recv_probs, num_local_experts, expert_counts)
        if built is not None:
            return built
        if not _ACE_TE_FELL_BACK:
            _ACE_TE_FELL_BACK.append(True)
            logger.info_rank0("ACE compaction: routing is not a multi-hot map, using the PyTorch path")
    return _torch_compact_permute(recv_hidden, recv_indices, recv_probs, num_local_experts, expert_counts)


def _compact_unpermute(expert_outputs, probs, token_rows, recv_tokens, row_map=None, width=None):
    weighted = expert_outputs * probs.to(expert_outputs.dtype).unsqueeze(-1)
    if token_rows is None:
        # TE layout: ``row_map`` is the ``[width, recv_tokens]`` expert-major map.
        return _TECompactUnpermute.apply(weighted, row_map, width, recv_tokens)
    # Accumulate in FP32, exactly as the combine does (see ``unpermute`` in
    # ``moe_utils.py``): summing the top-k contributions in BF16 costs ~1e-2 of
    # the sum's precision and, because ``index_add_`` reduces with atomics, the
    # result then depends on the (unstable) addition order.  Measured on
    # Qwen3.5-35B-A3B / 8x MTT S5000 this was the dominant run-to-run drift of
    # the ACE path: step-1 loss moved by 3.4e-4 and language-model gradients by
    # 4.5% between two byte-identical runs, versus 2e-8 / 0.2% with the FP32
    # accumulator.
    # The FP32 route materializes an upcast copy of ``weighted`` and an FP32
    # destination, so this fallback costs about 6x the transient memory of a
    # BF16 in-place add -- that is the price of the accuracy above.  The TE path
    # gets both for free.
    acc_dtype = torch.float32 if weighted.dtype in (torch.bfloat16, torch.float16) else weighted.dtype
    restored = torch.zeros(
        (recv_tokens, expert_outputs.shape[-1]),
        dtype=acc_dtype,
        device=expert_outputs.device,
    )
    restored.index_add_(0, token_rows, weighted.to(acc_dtype))
    return restored.to(expert_outputs.dtype)


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
    # This is immediately after the host-side dispatch call returns.  It must
    # stay outside the custom autograd Function: its forward executes with
    # grad recording disabled, while the shared expert needs a normal graph for
    # parameter and input gradients.
    # The received payload is consumed by local compaction below.
    invocation.wait_dispatch()
    num_local_experts = num_experts // state.ep_group.size()
    permuted, probs, token_rows, counts, row_map, width = _compact_permute(
        recv_hidden,
        recv_indices,
        recv_probs,
        num_local_experts,
        invocation.recv_counts,
    )
    cumsum = counts.cumsum(0)
    expert_outputs = ep_class.apply(permuted, cumsum, *ep_class_args)
    restored = _compact_unpermute(
        expert_outputs,
        probs,
        token_rows,
        recv_hidden.shape[0],
        row_map=row_map,
        width=width,
    )
    output = _ACECombine.apply(restored, invocation)
    if overlap is not None:
        # The shared expert ran on this stream while the dispatch was in flight,
        # so `finish` is a plain read; the add is ordered after both branches.
        output = output + overlap.finish()
    return output
