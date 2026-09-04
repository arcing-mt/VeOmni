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

"""MUSA-only runtime patch for Qwen3.5 text RoPE.

The current loader shares Qwen3.5's generated GPU module between CUDA and
MUSA. This module is intentionally a small runtime installer, not a patchgen
``PatchConfig``: it leaves the GPU/NPU generated files untouched and adds one
partial-RoPE OpSlot only on the active MUSA model path.
"""

from __future__ import annotations

from functools import wraps
from types import ModuleType

from ....ops.dispatch import OpSlot


def install_qwen3_5_musa_rotary_patch(modeling_module: ModuleType) -> None:
    """Install the MUSA text-RoPE OpSlot on one generated modeling module."""
    if getattr(modeling_module, "_VEOMNI_MUSA_ROTARY_PATCHED", False):
        return

    original = modeling_module.apply_rotary_pos_emb
    slot = OpSlot("rotary_pos_emb", "partial")

    @wraps(original)
    def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        if slot.use_non_eager_impl:
            return slot(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
        return original(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

    # _bind_veomni_ops() scans module globals for OpSlot instances after the
    # model class is selected. Adding the slot here keeps the CUDA generated
    # artifact unchanged while allowing the MUSA backend to bind and perform
    # the normal hardware check.
    modeling_module.veomni_apply_rotary_pos_emb = slot
    modeling_module.apply_rotary_pos_emb = apply_rotary_pos_emb
    modeling_module._VEOMNI_MUSA_ROTARY_PATCHED = True


__all__ = ["install_qwen3_5_musa_rotary_patch"]
