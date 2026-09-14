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

"""MUSA-only runtime patches for Qwen3.5.

The current loader shares Qwen3.5's generated GPU module between CUDA and
MUSA. This module is intentionally a small runtime installer, not a patchgen
``PatchConfig``: it leaves the GPU/NPU generated files untouched and adds one
partial/full RoPE OpSlots only on the active MUSA model path.

The MUSA chunk gated delta-rule backend does not need another runtime wrapper:
the shared generated module already exposes
``OpSlot("chunk_gated_delta_rule", "standard")``. At model-build time,
``_bind_veomni_ops`` resolves ``chunk_gated_delta_rule_implementation=musa``
through the normal kernel registry. Keeping that selection in the ops layer
avoids duplicating Qwen3.5's GatedDeltaNet forward here.
"""

from __future__ import annotations

from functools import wraps
from types import ModuleType

import torch

from ....ops.dispatch import OpSlot


class _MusaRotaryPhase:
    """Opaque phase carrier that FSDP mixed-precision casting leaves intact."""

    __slots__ = ("phase", "attention_scaling")

    def __init__(self, phase: torch.Tensor, attention_scaling: float) -> None:
        self.phase = phase
        self.attention_scaling = attention_scaling


def install_qwen3_5_musa_rotary_patch(
    modeling_module: ModuleType,
    *,
    install_text: bool = True,
    install_vision: bool = False,
) -> None:
    """Install requested MUSA RoPE OpSlots on one generated module.

    The two switches keep text and Vision configuration independent: a text
    model does not acquire a Vision slot, and a caller selecting only Vision
    MUSA does not alter the text path.
    """
    if install_text and not getattr(modeling_module, "_VEOMNI_MUSA_TEXT_ROTARY_PATCHED", False):
        original_text = modeling_module.apply_rotary_pos_emb
        text_slot = OpSlot("rotary_pos_emb", "partial")

        @wraps(original_text)
        def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
            if text_slot.use_non_eager_impl:
                # In the MUSA mode the rotary module returns (phase, scale),
                # while the generated attention code still names them cos/sin.
                if isinstance(cos, _MusaRotaryPhase):
                    return text_slot(
                        q,
                        k,
                        cos.phase,
                        attention_scaling=cos.attention_scaling,
                        unsqueeze_dim=unsqueeze_dim,
                    )
                return text_slot(q, k, cos, attention_scaling=sin, unsqueeze_dim=unsqueeze_dim)
            return original_text(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

        modeling_module.veomni_apply_rotary_pos_emb = text_slot
        modeling_module.apply_rotary_pos_emb = apply_rotary_pos_emb
        modeling_module._VEOMNI_MUSA_TEXT_ROTARY_PATCHED = True

        rotary_cls = getattr(modeling_module, "Qwen3_5MoeTextRotaryEmbedding", None) or getattr(
            modeling_module, "Qwen3_5TextRotaryEmbedding", None
        )
        if rotary_cls is not None and not getattr(modeling_module, "_VEOMNI_MUSA_TEXT_PHASE_PATCHED", False):
            @torch.no_grad()
            @modeling_module.dynamic_rope_update
            def rotary_forward(self, x, position_ids):
                """Return phase directly for MUSA; retain HF cos/sin for eager."""
                if position_ids.ndim == 2:
                    position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
                inv_freq_expanded = (
                    self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1).to(x.device)
                )
                position_ids_expanded = position_ids[:, :, None, :].float()
                device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
                with modeling_module.maybe_autocast(device_type=device_type, enabled=False):
                    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
                    freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
                    phase = torch.cat((freqs, freqs), dim=-1).contiguous()
                if text_slot.use_non_eager_impl:
                    # FSDP2 recursively casts tensors in forward inputs. A
                    # plain Python carrier keeps this FP32 phase opaque while
                    # preserving the generated model's two-item tuple API.
                    return _MusaRotaryPhase(phase, self.attention_scaling), None
                cos = phase.cos() * self.attention_scaling
                sin = phase.sin() * self.attention_scaling
                return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

            rotary_cls.forward = rotary_forward
            modeling_module._VEOMNI_MUSA_TEXT_PHASE_PATCHED = True

    if install_vision and not getattr(modeling_module, "_VEOMNI_MUSA_VISION_ROTARY_PATCHED", False):
        original_vision = modeling_module.apply_rotary_pos_emb_vision
        vision_slot = OpSlot("rotary_pos_emb_vision", "full")

        @wraps(original_vision)
        def apply_rotary_pos_emb_vision(q, k, cos, sin):
            if vision_slot.use_non_eager_impl:
                return vision_slot(q, k, cos, sin)
            return original_vision(q, k, cos, sin)

        modeling_module.veomni_apply_rotary_pos_emb_vision = vision_slot
        modeling_module.apply_rotary_pos_emb_vision = apply_rotary_pos_emb_vision
        modeling_module._VEOMNI_MUSA_VISION_ROTARY_PATCHED = True

    modeling_module._VEOMNI_MUSA_ROTARY_PATCHED = bool(
        getattr(modeling_module, "_VEOMNI_MUSA_TEXT_ROTARY_PATCHED", False)
        or getattr(modeling_module, "_VEOMNI_MUSA_VISION_ROTARY_PATCHED", False)
    )


__all__ = ["install_qwen3_5_musa_rotary_patch"]
