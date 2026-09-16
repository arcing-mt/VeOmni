import pytest
import torch

from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.distributed.moe.deepep_ace import (
    _ACEState,
    _compact_permute,
    _compact_unpermute,
    _load_deepep,
)


def test_deepep_ace_compact_round_trip_cpu():
    recv_hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    recv_indices = torch.tensor([[1, 0], [-1, 1], [0, -1]], dtype=torch.int64)
    recv_probs = torch.tensor([[0.25, 0.75], [0.0, 1.0], [1.0, 0.0]])

    permuted, probs, rows, counts = _compact_permute(recv_hidden, recv_indices, recv_probs, num_local_experts=2)
    _, _, _, host_counts = _compact_permute(
        recv_hidden, recv_indices, recv_probs, num_local_experts=2, expert_counts=[2, 2]
    )
    restored = _compact_unpermute(permuted, probs, rows, recv_hidden.shape[0])

    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert counts.tolist() == [2, 2]
    assert host_counts.tolist() == [2, 2]
    assert torch.equal(restored, expected)


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
