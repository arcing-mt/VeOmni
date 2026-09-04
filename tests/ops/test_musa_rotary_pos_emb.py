# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the MUSA RoPE registry adapter."""

from types import ModuleType

import pytest
import torch

import veomni.ops  # noqa: F401 - trigger kernel registrations
from veomni.models.transformers.qwen3_5.qwen3_5_musa_runtime_patch import install_qwen3_5_musa_rotary_patch
from veomni.ops.kernel_registry import KERNEL_REGISTRY
from veomni.ops.kernels.rotary.musa import _phase_from_cos_sin
from veomni.utils.device import IS_MUSA_AVAILABLE


def test_musa_rope_kernels_are_registered() -> None:
    assert "musa" in KERNEL_REGISTRY.list_available("rotary_pos_emb", "full")
    assert "musa" in KERNEL_REGISTRY.list_available("rotary_pos_emb", "partial")


def test_phase_reconstruction_marks_only_explicit_single_batch_as_shared() -> None:
    angles = torch.randn(2, 4, 8)
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1)
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1)
    phase, shared = _phase_from_cos_sin(cos, sin)
    assert not shared
    assert phase.shape == cos.shape
    assert torch.allclose(phase.cos(), cos, atol=1e-6, rtol=1e-6)
    assert torch.allclose(phase.sin(), sin, atol=1e-6, rtol=1e-6)

    shared_phase, is_shared = _phase_from_cos_sin(cos[:1], sin[:1])
    assert is_shared
    assert shared_phase.shape == (1, 4, 16)


def test_musa_patch_is_runtime_only_and_preserves_eager_when_unbound() -> None:
    module = ModuleType("fake_qwen3_5_modeling")

    def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        del unsqueeze_dim
        return q + cos, k + sin

    module.apply_rotary_pos_emb = apply_rotary_pos_emb
    install_qwen3_5_musa_rotary_patch(module)
    assert hasattr(module, "veomni_apply_rotary_pos_emb")
    assert getattr(module, "_VEOMNI_MUSA_ROTARY_PATCHED") is True

    q = torch.zeros(1)
    k = torch.ones(1)
    cos = torch.full((1,), 2.0)
    sin = torch.full((1,), 3.0)
    q_out, k_out = module.apply_rotary_pos_emb(q, k, cos, sin)
    assert torch.equal(q_out, q + cos)
    assert torch.equal(k_out, k + sin)


@pytest.mark.skipif(not IS_MUSA_AVAILABLE, reason="MUSA RoPE requires an active torch-musa device")
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_musa_partial_rope_matches_eager_and_preserves_tail(batch_size: int) -> None:
    torch.manual_seed(7110 + batch_size)
    device = torch.device("musa")
    batch, heads, seq, head_dim, rotary_dim = batch_size, 8, 32, 128, 64
    q = torch.randn(batch, heads, seq, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch, 4, seq, head_dim, device=device, dtype=torch.float16)
    angles = torch.randn(batch, seq, rotary_dim // 2, device=device, dtype=torch.float32)
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1).to(q.dtype)
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1).to(q.dtype)

    def eager(x: torch.Tensor) -> torch.Tensor:
        x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
        first, second = x_rot.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return torch.cat((x_rot * cos.unsqueeze(1) + rotated * sin.unsqueeze(1), x_pass), dim=-1)

    from veomni.ops.dispatch import OpSlot

    slot = OpSlot("rotary_pos_emb", "partial")
    slot.bind("musa")
    q_out, k_out = slot(q, k, cos, sin)
    q_ref, k_ref = eager(q), eager(k)
    torch.musa.synchronize()
    diff = torch.cat(((q_out.float() - q_ref.float()).abs().flatten(), (k_out.float() - k_ref.float()).abs().flatten()))
    assert float(diff.max().cpu()) <= 4e-3
    assert torch.equal(q_out[..., rotary_dim:], q[..., rotary_dim:])
    assert torch.equal(k_out[..., rotary_dim:], k[..., rotary_dim:])
