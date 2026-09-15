import torch
import pytest

from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.distributed.moe.deepep_ace import (
    _compact_permute,
    _compact_unpermute,
    _load_deepep,
)


def test_deepep_ace_compact_round_trip_cpu():
    recv_hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    recv_indices = torch.tensor([[1, 0], [-1, 1], [0, -1]], dtype=torch.int64)
    recv_probs = torch.tensor([[0.25, 0.75], [0.0, 1.0], [1.0, 0.0]])

    permuted, probs, rows, counts = _compact_permute(
        recv_hidden, recv_indices, recv_probs, num_local_experts=2
    )
    _, _, _, host_counts = _compact_permute(
        recv_hidden, recv_indices, recv_probs, num_local_experts=2, expert_counts=[2, 2]
    )
    restored = _compact_unpermute(permuted, probs, rows, recv_hidden.shape[0])

    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert counts.tolist() == [2, 2]
    assert host_counts.tolist() == [2, 2]
    assert torch.equal(restored, expected)


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
