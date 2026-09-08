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

"""MUSA-native fused rotary positional embedding implementations.

The torch-musa RoPE binding is backed by muDNN and takes a float32 phase
table, rather than the ``cos``/``sin`` pair exposed by HuggingFace model
functions.  The helpers below convert the model representation once per
``cos`` tensor (the same position embeddings are reused by every layer), then
choose the muDNN layout according to what is known about the phase batch:

* ``[1, S, D]`` is explicitly batch-shared and uses the native batch-first
  layout ``[B, S, H, D]``.
* ``[B, S, D]`` is conservatively treated as batch-specific and flattens to
  ``[B*S, 1, H, D]`` so MRoPE/padding/packing phases are preserved without a
  device-to-host equality check.

Only the half-split (GPT-NeoX/HuggingFace) layout is exposed here.  The
interleaved layout has a different phase packing and is not silently mapped
to this backend.
"""

from __future__ import annotations

import weakref
from typing import NamedTuple

import torch

from ....distributed.parallel_state import get_parallel_state, is_parallel_state_initialized


class _PhaseCacheEntry(NamedTuple):
    cos_ref: weakref.ReferenceType[torch.Tensor]
    sin_ref: weakref.ReferenceType[torch.Tensor]
    phase: torch.Tensor
    shared: bool
    rotary_dim: int


# ``cos``/``sin`` are created once by the model and reused by all decoder
# layers.  Cache the angle conversion without retaining old forward passes or
# introducing a per-layer atan2 launch.  Tensor ids are guarded by weakrefs so
# id reuse cannot return a stale phase table.
_PHASE_CACHE: dict[int, _PhaseCacheEntry] = {}


def _phase_from_cos_sin(cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Recover a muDNN float32 phase table from HF cos/sin tensors."""
    if cos.ndim not in (2, 3) or sin.ndim != cos.ndim:
        raise ValueError(
            "MUSA RoPE expects cos and sin with shape [sequence, rotary_dim] or "
            "[batch, sequence, rotary_dim]. "
            f"Got cos.ndim={cos.ndim}, sin.ndim={sin.ndim}."
        )
    if cos.shape != sin.shape:
        raise ValueError(f"MUSA RoPE requires cos and sin to have the same shape; got {cos.shape} and {sin.shape}.")
    rotary_dim = cos.shape[-1]
    if rotary_dim == 0 or rotary_dim % 2:
        raise ValueError(f"MUSA RoPE requires an even, non-zero rotary dimension; got {rotary_dim}.")

    key = id(cos)
    cached = _PHASE_CACHE.get(key)
    if (
        cached is not None
        and cached.cos_ref() is cos
        and cached.sin_ref() is sin
        and cached.rotary_dim == rotary_dim
        and cached.phase.shape == cos.shape
    ):
        return cached.phase, cached.shared

    # HF builds cos/sin as [cos(theta_0..theta_n), cos(theta_0..theta_n)]
    # and the analogous sin table.  muDNN wants the corresponding angle table
    # in float32.  atan2 is done only once for the shared position tensors and
    # the result is reused by all layers.
    half = rotary_dim // 2
    phase_half = torch.atan2(sin[..., :half].float(), cos[..., :half].float())
    phase = torch.cat((phase_half, phase_half), dim=-1).contiguous()
    # A vision call supplies [S, D] and is inherently one shared sequence;
    # text supplies [B, S, D], where only B==1 is known to be shared without
    # comparing device tensors.
    shared = cos.ndim == 2 or cos.shape[0] == 1

    entry: _PhaseCacheEntry

    def _remove(
        _ref: weakref.ReferenceType[torch.Tensor], *, cache_key: int = key, cache: dict[int, _PhaseCacheEntry] = _PHASE_CACHE
    ) -> None:
        if cache.get(cache_key) is entry:
            cache.pop(cache_key, None)

    entry = _PhaseCacheEntry(
        cos_ref=weakref.ref(cos, _remove),
        sin_ref=weakref.ref(sin),
        phase=phase,
        shared=shared,
        rotary_dim=rotary_dim,
    )
    _PHASE_CACHE[key] = entry
    return phase, shared


def _rope_one(x: torch.Tensor, phase: torch.Tensor, shared: bool) -> torch.Tensor:
    """Apply muDNN RoPE to one ``[B, H, S, D]`` tensor."""
    if x.ndim != 4:
        raise ValueError(f"MUSA RoPE expects q/k with shape [B, H, S, D]; got {tuple(x.shape)}.")
    batch_size, _, sequence_length, rotary_dim = x.shape
    if phase.shape[1] != sequence_length:
        raise ValueError(
            "MUSA RoPE phase sequence length does not match q/k: "
            f"phase={phase.shape[1]}, input={sequence_length}."
        )
    if shared:
        # Explicitly shared phase: muDNN consumes [S, D] for a [B, S, H, D]
        # batch-first input.
        x_bshd = x.transpose(1, 2).contiguous()
        return torch.rope(x_bshd, phase[0], False, True, False).transpose(1, 2)

    if phase.shape[0] != batch_size:
        raise ValueError(
            "MUSA RoPE received a batch-specific phase table with a different batch size: "
            f"phase={phase.shape[0]}, input={batch_size}."
        )
    # Batch-specific phase: flatten B*S into muDNN's sequence axis and keep a
    # singleton batch axis.  This is one launch per q/k tensor, not one launch
    # per sample, while preserving each sample's MRoPE phase.
    x_flat = x.transpose(1, 2).reshape(batch_size * sequence_length, 1, x.shape[1], rotary_dim)
    phase_flat = phase.reshape(batch_size * sequence_length, rotary_dim)
    out_flat = torch.rope(x_flat, phase_flat, False, False, False)
    return out_flat.reshape(batch_size, sequence_length, x.shape[1], rotary_dim).transpose(1, 2)


def _apply_rotary_pos_emb_musa(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int = 1,
    partial: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if unsqueeze_dim != 1:
        raise ValueError(
            "MUSA RoPE currently supports the HuggingFace [B, H, S, D] calling convention "
            "with unsqueeze_dim=1 only."
        )
    phase, shared = _phase_from_cos_sin(cos, sin)
    rotary_dim = phase.shape[-1]
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError(f"MUSA RoPE expects q/k with shape [B, H, S, D]; got q={tuple(q.shape)}, k={tuple(k.shape)}.")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError(f"MUSA RoPE requires q/k batch and sequence dimensions to match; got q={q.shape}, k={k.shape}.")
    if not partial and (q.shape[-1] != rotary_dim or k.shape[-1] != rotary_dim):
        raise ValueError(
            "MUSA full RoPE requires rotary_dim to equal q/k head_dim; "
            f"got rotary_dim={rotary_dim}, q_dim={q.shape[-1]}, k_dim={k.shape[-1]}."
        )
    if partial and (rotary_dim > q.shape[-1] or rotary_dim > k.shape[-1]):
        raise ValueError(
            "MUSA partial RoPE rotary_dim must not exceed q/k head_dim; "
            f"got rotary_dim={rotary_dim}, q_dim={q.shape[-1]}, k_dim={k.shape[-1]}."
        )

    q_rot, k_rot = q[..., :rotary_dim], k[..., :rotary_dim]
    q_embed = _rope_one(q_rot, phase, shared)
    k_embed = _rope_one(k_rot, phase, shared)
    if not partial:
        return q_embed, k_embed
    return torch.cat((q_embed, q[..., rotary_dim:]), dim=-1), torch.cat((k_embed, k[..., rotary_dim:]), dim=-1)


def apply_rotary_pos_emb_musa(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MUSA full RoPE adapter for the standard HF ``(q, k, cos, sin)`` API."""
    del position_ids
    return _apply_rotary_pos_emb_musa(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim, partial=False)


def partial_apply_rotary_pos_emb_musa(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MUSA partial RoPE adapter used by Qwen3.5 text attention."""
    return _apply_rotary_pos_emb_musa(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim, partial=True)


def apply_rotary_pos_emb_vision_musa(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MUSA full RoPE adapter for Qwen3.5 Vision's ``[S, H, D]`` API."""
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError(f"MUSA Vision RoPE expects q/k with shape [S, H, D]; got q={tuple(q.shape)}, k={tuple(k.shape)}.")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError(f"MUSA Vision RoPE requires q/k sequence and head dimensions to match; got q={q.shape}, k={k.shape}.")
    phase, _ = _phase_from_cos_sin(cos, sin)
    if phase.ndim != 2 or phase.shape[0] != q.shape[0] or phase.shape[1] != q.shape[-1]:
        raise ValueError(
            "MUSA Vision RoPE requires cos/sin with shape [S, D] matching q/k; "
            f"got phase={tuple(phase.shape)}, q={tuple(q.shape)}."
        )
    # muDNN's non-batch-first contract is [S, B, H, D].  Use B=1 because
    # Vision tokens are already packed along S and their phase is per token.
    q_out = torch.rope(q.unsqueeze(1).contiguous(), phase, False, False, False).squeeze(1)
    k_out = torch.rope(k.unsqueeze(1).contiguous(), phase, False, False, False).squeeze(1)

    # The generated Vision path only introduces zero cos/sin rows when
    # sequence parallelism is enabled (``sp_pad_and_slice``).  Avoid an extra
    # device-side mask kernel on the common non-SP path.  Eager RoPE zeros
    # those rows, while atan2 above maps them to phase=0 (identity rotation),
    # so restore the eager result only for the SP path.
    if is_parallel_state_initialized() and get_parallel_state().sp_enabled:
        valid = ((cos != 0).any(dim=-1) | (sin != 0).any(dim=-1)).view(-1, 1, 1)
        q_out = torch.where(valid, q_out, torch.zeros_like(q_out))
        k_out = torch.where(valid, k_out, torch.zeros_like(k_out))
    return q_out, k_out


__all__ = [
    "apply_rotary_pos_emb_musa",
    "apply_rotary_pos_emb_vision_musa",
    "partial_apply_rotary_pos_emb_musa",
]
