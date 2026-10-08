"""Routing validation must not reject distinct received expert slots."""

import pytest
import torch

from veomni.distributed.moe import deepep_ace as ace


DEVICES = [
    "cpu",
    pytest.param(
        "musa",
        marks=pytest.mark.skipif(not hasattr(torch, "musa") or not torch.musa.is_available(), reason="requires MUSA"),
    ),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("num_tokens", [0, 1, 8192, 27597, 32640, 32768])
def test_distinct_routes_pass_te_validation(monkeypatch, device, num_tokens):
    width, top_k = 32, 8
    flat = torch.arange(num_tokens * top_k).view(num_tokens, top_k)
    indices = flat % width
    indices[flat % 3 == 0] = -1
    expected = torch.bincount(indices[indices >= 0], minlength=width)
    hidden = torch.empty(num_tokens, 16, device=device, dtype=torch.bfloat16)
    probabilities = torch.ones(num_tokens, top_k, device=device)
    # Isolate the real routing guards; TE payload kernels are tested separately.
    monkeypatch.setattr(ace, "_te_permute_with_plan", lambda _hidden, _probs, plan: plan)
    result = ace._te_compact_permute(hidden, indices.to(device), probabilities, width, expected.tolist())
    assert result is not None, "Distinct routes with matching DeepEP counts must use the TE path"
    torch.testing.assert_close(result["counts"].cpu(), expected, rtol=0, atol=0)
    assert result["num_out"] == int(expected.sum())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("invalid", ["duplicate", "expert_counts"])
def test_te_validation_still_rejects_invalid_routing(device, invalid):
    indices = torch.tensor([[0, 1, -1, -1], [1, 2, -1, -1]], device=device)
    counts = [1, 2, 1, 0]
    if invalid == "duplicate":
        indices[0, 1] = 0
    else:
        counts[0] += 1
    result = ace._te_compact_permute(
        torch.empty(2, 16, device=device, dtype=torch.bfloat16),
        indices,
        torch.ones(2, 4, device=device),
        4,
        counts,
    )
    assert result is None
