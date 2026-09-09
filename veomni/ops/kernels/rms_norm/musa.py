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

"""MUSA-native fused RMSNorm implementations.

``torch.rms_norm`` dispatches to torch-musa's ``aten::_fused_rms_norm``
implementation on MUSA.  The explicit functional helpers below keep the
variant semantics visible to the registry: standard RMSNorm uses ``weight``
as-is, while Qwen3.5 uses its zero-centred ``1 + weight`` scale.
"""

from __future__ import annotations

import torch


def _musa_rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """Call PyTorch's native RMSNorm API (fused by torch-musa on MUSA)."""
    return torch.rms_norm(hidden_states, [hidden_states.shape[-1]], weight, eps)


def standard_rms_norm_forward_musa(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """MUSA fused RMSNorm for the standard ``weight * x/rms`` variant."""
    input_dtype = hidden_states.dtype
    compute_dtype = torch.promote_types(input_dtype, weight.dtype)
    if hidden_states.dtype != compute_dtype:
        hidden_states = hidden_states.to(compute_dtype)
    if weight.dtype != compute_dtype:
        weight = weight.to(compute_dtype)
    return _musa_rms_norm(hidden_states, weight, eps).to(input_dtype)


def unweighted_rms_norm_forward_musa(
    hidden_states: torch.Tensor, weight: torch.Tensor | None, eps: float
) -> torch.Tensor:
    """MUSA fused non-affine RMSNorm for DeepSeek-V4's unweighted variant."""
    if weight is not None:
        raise ValueError("Unweighted RMSNorm expects weight=None.")
    return _musa_rms_norm(hidden_states, None, eps)


def qwen3_5_rms_norm_forward_musa(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """MUSA fused RMSNorm for Qwen3.5's ``(1 + weight) * x/rms`` variant."""
    input_dtype = hidden_states.dtype
    compute_dtype = torch.promote_types(input_dtype, weight.dtype)
    hidden_states = hidden_states.to(compute_dtype)
    scale = (1.0 + weight.to(compute_dtype)).to(compute_dtype)
    return _musa_rms_norm(hidden_states, scale, eps).to(input_dtype)


def rms_norm_forward_musa(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Class-forward adapter used by legacy ``apply_per_model_patches``."""
    eps = getattr(self, "variance_epsilon", getattr(self, "eps", None))
    if eps is None:
        raise AttributeError("MUSA RMSNorm adapter requires an `eps` or `variance_epsilon` attribute.")
    return standard_rms_norm_forward_musa(hidden_states, self.weight, eps)


__all__ = [
    "qwen3_5_rms_norm_forward_musa",
    "rms_norm_forward_musa",
    "standard_rms_norm_forward_musa",
    "unweighted_rms_norm_forward_musa",
]
