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

"""Numerical tests for the torch-musa fused RMSNorm registry backend."""

import pytest
import torch

import veomni.ops  # noqa: F401 - trigger kernel registrations
from veomni.ops.dispatch import OpSlot
from veomni.utils.device import IS_MUSA_AVAILABLE


pytestmark = pytest.mark.skipif(not IS_MUSA_AVAILABLE, reason="MUSA RMSNorm requires an active torch-musa device")


def _eager_standard(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_f = x.float()
    normed = x_f * torch.rsqrt(x_f.square().mean(-1, keepdim=True) + eps)
    return (weight * normed.to(x.dtype)).to(x.dtype)


def _eager_qwen3_5(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_f = x.float()
    normed = x_f * torch.rsqrt(x_f.square().mean(-1, keepdim=True) + eps)
    return ((1.0 + weight.float()) * normed).to(x.dtype)


@pytest.mark.parametrize("shape", [(2, 16, 128), (1, 8, 4096)])
def test_musa_standard_rms_norm_matches_eager(shape: tuple[int, ...]) -> None:
    torch.manual_seed(6101)
    eps = 1e-6
    x = torch.randn(shape, device="musa", dtype=torch.bfloat16).contiguous()
    weight = torch.randn(shape[-1], device="musa", dtype=torch.bfloat16).contiguous()
    slot = OpSlot("rms_norm", "standard")
    slot.bind("musa")

    output = slot(x, weight, eps)
    reference = _eager_standard(x, weight, eps)
    assert torch.allclose(output, reference, atol=1e-2, rtol=1e-2)

    grad_output = torch.randn_like(output)
    x_kernel = x.detach().clone().requires_grad_()
    w_kernel = weight.detach().clone().requires_grad_()
    slot(x_kernel, w_kernel, eps).backward(grad_output)
    x_ref = x.detach().clone().requires_grad_()
    w_ref = weight.detach().clone().requires_grad_()
    _eager_standard(x_ref, w_ref, eps).backward(grad_output)
    # The native MUSA BF16 backward reduces the weight gradient in a different
    # order from the eager graph.  Keep a meaningful relative gate while
    # allowing the observed worst-case BF16 ULP accumulation in the reduction.
    assert torch.allclose(x_kernel.grad, x_ref.grad, atol=1e-2, rtol=1e-2)
    assert torch.allclose(w_kernel.grad, w_ref.grad, atol=0.25, rtol=2e-2)


def test_musa_qwen3_5_rms_norm_matches_eager() -> None:
    torch.manual_seed(6102)
    eps = 1e-6
    x = torch.randn(2, 16, 128, device="musa", dtype=torch.bfloat16).contiguous()
    weight = (0.01 * torch.randn(128, device="musa", dtype=torch.bfloat16)).contiguous()
    slot = OpSlot("rms_norm", "qwen3_5")
    slot.bind("musa")

    output = slot(x, weight, eps)
    reference = _eager_qwen3_5(x, weight, eps)
    assert torch.allclose(output, reference, atol=1e-2, rtol=1e-2)


def test_musa_qwen3_5_rms_norm_mixed_dtype_matches_eager() -> None:
    torch.manual_seed(6104)
    eps = 1e-6
    x = torch.randn(32, 128, device="musa", dtype=torch.bfloat16).contiguous()
    weight = (0.5 * torch.randn(128, device="musa", dtype=torch.float32)).contiguous()
    slot = OpSlot("rms_norm", "qwen3_5")
    slot.bind("musa")

    output = slot(x, weight, eps)
    reference = _eager_qwen3_5(x, weight, eps)
    torch.testing.assert_close(output, reference, rtol=2e-3, atol=2e-3)


def test_musa_unweighted_rms_norm_matches_eager() -> None:
    torch.manual_seed(6103)
    eps = 1e-6
    x = torch.randn(2, 16, 128, device="musa", dtype=torch.bfloat16).contiguous()
    slot = OpSlot("rms_norm", "unweighted")
    slot.bind("musa")

    output = slot(x, None, eps)
    reference = x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps).to(x.dtype)
    assert torch.allclose(output, reference, atol=1e-2, rtol=1e-2)
