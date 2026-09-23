from types import SimpleNamespace

import pytest
import torch

from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.distributed.moe.deepep_ace import (
    _ACTIVE_SHARED_EXPERT,
    _ACEState,
    _compact_permute,
    _compact_unpermute,
    _load_deepep,
    _SharedExpertOverlap,
)


def test_deepep_ace_compact_round_trip_cpu():
    recv_hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    recv_indices = torch.tensor([[1, 0], [-1, 1], [0, -1]], dtype=torch.int64)
    recv_probs = torch.tensor([[0.25, 0.75], [0.0, 1.0], [1.0, 0.0]])

    permuted, probs, rows, counts, row_map, width = _compact_permute(
        recv_hidden, recv_indices, recv_probs, num_local_experts=2
    )
    _, _, _, host_counts, _, _ = _compact_permute(
        recv_hidden, recv_indices, recv_probs, num_local_experts=2, expert_counts=[2, 2]
    )
    restored = _compact_unpermute(permuted, probs, rows, recv_hidden.shape[0], row_map=row_map, width=width)

    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert counts.tolist() == [2, 2]
    assert host_counts.tolist() == [2, 2]
    assert torch.equal(restored, expected)
    # The CPU path is driven by ``token_rows`` (row -> received token) with the
    # top-k slot count as its width; the expert-major map is a TE-path artifact.
    assert row_map is None and width == 2
    assert rows.tolist() == [0, 2, 0, 1]


def test_deepep_ace_compact_unpermute_backward_cpu():
    expert_outputs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    probs = torch.tensor([0.25, 0.5, 0.75], requires_grad=True)
    token_rows = torch.tensor([0, 0, 1], dtype=torch.long)

    restored = _compact_unpermute(expert_outputs, probs, token_rows, recv_tokens=2)
    restored.sum().backward()

    assert torch.equal(expert_outputs.grad, probs.detach().unsqueeze(-1).expand_as(expert_outputs))
    assert torch.equal(probs.grad, expert_outputs.detach().sum(dim=-1))


def test_deepep_ace_dispatch_uses_caller_previous_event(monkeypatch):
    previous_event = object()
    layout_event = object()

    class FakeGroup:
        def size(self):
            return 2

    class FakeBuffer:
        def get_dispatch_layout(self, *args, **kwargs):
            assert kwargs["previous_event"] is previous_event
            return None, None, None, None, layout_event

        def dispatch(self, hidden_states, **kwargs):
            assert kwargs["previous_event"] is layout_event
            return hidden_states, kwargs["topk_idx"], kwargs["topk_weights"], [1], object(), object()

    state = _ACEState(FakeGroup(), num_experts=2, top_k=1)
    monkeypatch.setattr(state, "_get_buffer", lambda _hidden_states: FakeBuffer())
    hidden_states = torch.ones(1, 2)
    selected_experts = torch.zeros(1, 1, dtype=torch.long)
    routing_weights = torch.ones(1, 1)

    received, _, _ = state.dispatch(hidden_states, selected_experts, routing_weights, previous_event=previous_event)
    assert received is hidden_states


def test_deepep_ace_is_a_dispatcher_selection():
    config = OpsImplementationConfig(moe_dispatcher="deepep_ace")
    assert config.moe_implementation == "fused_triton"
    assert config.moe_dispatcher == "deepep_ace"
    assert config.moe_deepep_num_sms == 20
    assert config.moe_deepep_token_capacity == 8192


def test_deepep_ace_num_sms_must_be_even_and_positive():
    with pytest.raises(ValueError, match="positive even"):
        OpsImplementationConfig(moe_deepep_num_sms=3)


def test_deepep_ace_token_capacity_must_be_positive():
    with pytest.raises(ValueError, match="token_capacity must be positive"):
        OpsImplementationConfig(moe_deepep_token_capacity=0)


def test_deepep_wheel_event_exports_are_resolved():
    _, event_handle, event_overlap = _load_deepep()
    assert event_handle is not None
    assert event_overlap is not None


def test_shared_expert_overlap_runs_inline_and_joins(monkeypatch):
    """The overlap must gate and compute the shared expert without a stream of its own.

    An earlier revision queued it on a dedicated MUSA stream; that cost more
    than the shared expert itself (~0.7 s/step on Qwen3.5-35B-A3B).  Pin the
    contract as well as the numerics: no extra stream, no cross-stream event,
    and `finish` returns exactly what `start` computed.
    """
    stream = object()
    monkeypatch.setattr(torch, "musa", SimpleNamespace(current_stream=lambda: stream), raising=False)
    created_streams = []
    monkeypatch.setattr(torch, "Stream", lambda *a, **k: created_streams.append((a, k)), raising=False)

    hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    class FakeSharedExpert(torch.nn.Module):
        def forward(self, x):
            return x * 2.0

    class FakeGate(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 1)

    overlap = _SharedExpertOverlap(FakeSharedExpert(), FakeGate(), hidden_states)
    with pytest.raises(RuntimeError, match="was not started"):
        overlap.finish()

    overlap.start()
    expected = torch.sigmoid(torch.zeros(hidden_states.shape[0], 1)) * (hidden_states * 2.0)

    assert overlap.finish() is overlap.output
    assert torch.allclose(overlap.output, expected)
    assert overlap.stream is stream
    assert created_streams == []
    assert not hasattr(overlap, "event")

    with pytest.raises(RuntimeError, match="already started"):
        overlap.start()
    other = object()
    monkeypatch.setattr(torch, "musa", SimpleNamespace(current_stream=lambda: other), raising=False)
    with pytest.raises(RuntimeError, match="different stream"):
        overlap.finish()


def test_shared_expert_is_issued_between_dispatch_and_wait(monkeypatch):
    """The overlap only exists if the shared expert is issued inside the dispatch window.

    A shared expert issued before ``buffer.dispatch`` cannot hide under the
    payload; one issued after ``wait_dispatch`` does not overlap at all.  Both
    mistakes have happened in this file's history, so assert the order against
    a fake DeepEP buffer rather than trusting the call site.
    """
    order: list[str] = []

    class FakeEvent:
        def current_stream_wait(self):
            pass

    class FakeGroup:
        def size(self):
            return 2

    class FakeBuffer:
        def get_dispatch_layout(self, *args, **kwargs):
            order.append("layout")
            return None, None, None, None, FakeEvent()

        def dispatch(self, hidden_states, **kwargs):
            order.append("dispatch")
            return hidden_states, kwargs["topk_idx"], kwargs["topk_weights"], [1, 1], object(), FakeEvent()

    state = _ACEState(FakeGroup(), num_experts=2, top_k=1)
    monkeypatch.setattr(state, "_get_buffer", lambda _hidden_states: FakeBuffer())

    class FakeSharedExpert(torch.nn.Module):
        def forward(self, x):
            order.append("shared_expert")
            return x * 2.0

    class FakeGate(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 1)

    hidden_states = torch.tensor([[1.0, 2.0], [1.0, 1.0]])
    selected_experts = torch.tensor([[0], [1]])
    routing_weights = torch.ones(2, 1)
    overlap = _SharedExpertOverlap(FakeSharedExpert(), FakeGate(), hidden_states)

    def run_dispatch(active):
        order.clear()
        token = _ACTIVE_SHARED_EXPERT.set(active) if active is not None else None
        try:
            state.dispatch(hidden_states, selected_experts, routing_weights)
        finally:
            if token is not None:
                _ACTIVE_SHARED_EXPERT.reset(token)

    # Without an active overlap context the dispatcher must not run a shared expert.
    run_dispatch(None)
    assert order == ["layout", "dispatch"], order

    # With one, it is issued after the dispatch is submitted ...
    run_dispatch(overlap)
    assert order == ["layout", "dispatch", "shared_expert"], order

    # ... and before the payload is awaited: `wait_dispatch` must never observe a
    # missing shared expert, otherwise the overlap is silently gone.
    def wait():
        assert order[-1] == "shared_expert", f"shared expert not issued before wait_dispatch: {order}"
        order.append("wait_dispatch")
        return state.dispatch_event.current_stream_wait()

    monkeypatch.setattr(state, "wait_dispatch", wait)
    state.wait_dispatch()
    assert order[-1] == "wait_dispatch"
