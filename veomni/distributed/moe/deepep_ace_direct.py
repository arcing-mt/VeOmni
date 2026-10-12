"""MUSA TE producers writing directly into leased DeepEP ACE combine windows.

Imported only when model.ops_implementation.moe_deepep_combine_direct_write is
enabled. The pool owns two registered windows; each invocation borrows a window
only until its combine submission has an associated completion event.
"""

import os
import threading
from contextlib import contextmanager

import torch

from . import deepep_ace as ace
from . import deepep_ace_checkpoint as ckpt


class WindowLoan:
    """Keep per-call tensor references on an instance, never on a dynamic class."""

    def __init__(self, pool, state, slot, rows, with_weights, x, p):
        self.pool, self.state = pool, state
        self.slot, self.rows, self.with_weights = slot, rows, with_weights
        self.x, self.p = x, p
        self.submitted = False

    def submit(self, payload):
        if self.submitted or self.x is None or payload.data_ptr() != self.x.data_ptr():
            raise RuntimeError("ACE combine must consume the active window exactly once")
        try:
            result = self.pool.buffer.combine(
                self.x,
                self.state.handle,
                topk_weights=self.p,
                previous_event=ace._current_stream_event(),
                async_finish=True,
                allocate_on_comm_stream=True,
            )
            self.pool.events[self.slot] = result[-1]
            self.submitted = True
            result[-1].current_stream_wait()
            assert not self.rows or result[0].data_ptr() != self.x.data_ptr()
            return result[:2]
        except BaseException:
            self.pool.poisoned = True
            raise

    def release(self):
        self.x = self.p = self.state = self.pool = None


class WindowPool:
    def __init__(self, buffer):
        self.buffer = buffer
        self.lock = threading.Lock()
        self.events = [None] * 2
        self.next = 0
        self.poisoned = False

    @contextmanager
    def acquire(self, state, reference, rows, with_weights):
        with self.lock:
            if self.poisoned:
                raise RuntimeError("a previous window submission failed")
            slot = self.next
            self.next = (slot + 1) % 2
            if self.events[slot] is not None:
                self.events[slot].current_stream_wait()
            capacity, hidden, topk = self.buffer._veomni_combine_shape
            if (
                reference.ndim != 2
                or reference.dtype != torch.bfloat16
                or reference.shape[-1] != hidden
                or not 0 <= rows <= capacity * topk
            ):
                raise ValueError("unsupported direct-window payload")
            x, p = self.buffer.get_ace_combine_buffer(rows, hidden, topk, with_weights, buffer_index=slot)
            # Keep a non-null view on empty ranks: DeepEP's null-input fallback uses slot 0.
            if (
                x.shape != (max(rows, 1), hidden)
                or x.dtype != torch.bfloat16
                or x.device != reference.device
                or not x.is_contiguous()
            ):
                raise RuntimeError("DeepEP returned an incompatible activation window")
            if with_weights and (
                p is None
                or p.shape != (max(rows, 1), topk)
                or p.dtype != torch.float32
                or p.device != reference.device
                or not p.is_contiguous()
            ):
                raise RuntimeError("DeepEP returned an incompatible routing-gradient window")
            loan = WindowLoan(self, state, slot, rows, with_weights, x, p)
            try:
                yield loan
            finally:
                # Checkpoint early-stop may occur after the TE write but before combine.
                # Fence that producer before releasing its slot, without adding communication.
                try:
                    if not loan.submitted:
                        self.events[slot] = ace._current_stream_event()
                finally:
                    loan.release()

    def produce_and_combine(self, state, reference, rows, producer, with_weights):
        with self.acquire(state, reference, rows, with_weights) as loan:
            try:
                if rows:
                    producer(loan.x, loan.p)
                return loan.submit(loan.x)
            except BaseException:
                self.poisoned = True
                raise


class DirectState(ace._ACEState):
    loan = None

    def pool(self, reference):
        buffer = self._get_buffer(reference)
        if not hasattr(buffer, "_lease_pool"):
            buffer._lease_pool = WindowPool(buffer)
        return buffer._lease_pool

    def written_combine(self, reference, rows, producer, with_weights=False):
        return self.pool(reference).produce_and_combine(self, reference, rows, producer, with_weights)

    def combine(self, expert_outputs):
        if self.loan is not None:
            return self.loan.submit(expert_outputs)[0]
        return super().combine(expert_outputs)


class Unpermuted(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weighted, row_map, width, rows, loan):
        ctx.width, ctx.num_tokens, ctx.num_out = width, rows, weighted.shape[0]
        ctx.save_for_backward(row_map.t().contiguous())
        _, _, unpermute, dtype = ace._ace_te_impl()
        empty = torch.empty(0)
        if rows:
            result = unpermute(
                dtype[weighted.dtype], weighted, row_map, empty, empty, rows, width, weighted.shape[-1], loan.x, empty
            )[0]
            assert result.data_ptr() == loan.x.data_ptr()
        return loan.x

    @staticmethod
    def backward(ctx, grad):
        return (*ace._TECompactUnpermute.backward(ctx, grad), None)


def reverse_compact(state, grad, grad_probs, row_nt, row_map, scatter, rows, width):
    _, _, unpermute, dtype = ace._ace_te_impl()
    empty = torch.empty(0)

    def write(x, p):
        result = unpermute(
            dtype[grad.dtype], grad.contiguous(), row_map, empty, empty, rows, width, grad.shape[-1], x, empty
        )[0]
        assert result.data_ptr() == x.data_ptr()
        if grad_probs is not None:
            slot_gradients = None
            if os.environ.get("VEOMNI_MUSA_ACE_TE_SLOT_PROBABILITY", "0") == "1":
                from ...ops.kernels.moe.musa_te_slot_probability import slot_gradients as gather_slot_gradients

                slot_gradients = gather_slot_gradients(grad_probs, row_map, scatter, index_offset=1, out=p)
            if slot_gradients is None:
                dense = grad_probs[row_nt.clamp_min(0)] * (row_nt >= 0).to(grad_probs.dtype)
                torch.gather(dense, 1, (scatter - 1).clamp_min(0), out=p)
                p.masked_fill_(scatter == 0, 0)

    return state.written_combine(grad, rows, write, grad_probs is not None)


class Dispatched(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, selected, weights, state):
        received, indices, recv_probs = state.dispatch(
            hidden, selected, weights, getattr(state, "previous_event", None)
        )
        state.wait_dispatch()
        plan = {}
        compact, probs, rows, counts, row_map, width = ace._compact_permute(
            received, indices, recv_probs, state.num_experts // state.group.size(), state.recv_counts, plan=plan
        )
        if plan["backend"] != "te":
            raise RuntimeError("ACE direct write requires native TE compaction")
        ctx.state, ctx.rows, ctx.width = state, received.shape[0], width
        state.received_tokens = ctx.rows
        ctx.save_for_backward(row_map.t().contiguous(), row_map, plan["scatter_indices"])
        ctx.mark_non_differentiable(counts, row_map)
        return compact, probs, rows, counts, row_map, width

    @staticmethod
    def backward(ctx, grad, gp, *unused):
        row_nt, row_map, scatter = ctx.saved_tensors
        dx, dp = reverse_compact(ctx.state, grad, gp, row_nt, row_map, scatter, ctx.rows, ctx.width)
        return dx, None, dp, None


def dispatch_direct(ep_class, num_experts, weights, selected, hidden, *params):
    ps = ace.get_parallel_state()
    state = DirectState(ps.ep_group, num_experts, selected.shape[-1])
    if (
        hidden.dtype != torch.bfloat16
        or not getattr(hidden, "is_musa", False)
        or not ace._ace_te_supported(hidden, weights.float(), num_experts // ps.ep_group.size())
    ):
        raise ValueError("ACE direct write requires contiguous MUSA BF16 activations and TransformerEngine compaction")
    overlap = ace._ACTIVE_SHARED_EXPERT.get()
    if overlap is not None:
        state.previous_event = ace._current_stream_event()
    fn = ckpt._DispatchCompact if ckpt.has_cached_dispatch() else Dispatched
    compact, probs, rows, counts, row_map, width = fn.apply(hidden, selected, weights.float(), state)
    if row_map is None:
        raise RuntimeError("ACE direct write requires TE row maps")
    expert = ep_class.apply(compact, counts.cumsum(0), *params)
    weighted = expert * probs.to(expert.dtype).unsqueeze(-1)
    with state.pool(expert).acquire(state, expert, state.received_tokens, False) as loan:
        state.loan = loan
        try:
            restored = Unpermuted.apply(weighted, row_map, width, state.received_tokens, loan)
            result = ace._ACECombine.apply(restored, state)
        finally:
            state.loan = None
    return result + overlap.finish() if overlap is not None else result
