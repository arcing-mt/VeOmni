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

"""MUSA TileLang chunk gated delta rule: adapter over ``torch_kernels``.

``torch_kernels`` ships an end-to-end TileLang GDN (KKT+solve, WY recompute, state
scan, output, and the full backward), MUSA-native. At the production geometry
(``B=1``, ``S~8.5k`` packed varlen, ``H=32``, ``D=128``) its kernels are ~3x faster
than the tuned FLA Triton path, but **only if L2 normalization is kept out of its
adapter**: ``torch_kernels.gated_delta_net`` implements
``use_qk_l2norm_in_kernel=True`` as a *torch-level* chain
(``x.float().square().sum(-1).add(eps).sqrt().div()``), measured at 1.74 ms for
``q``+``k`` versus 0.44 ms for ``fla.modules.l2norm.l2norm`` -- that 4x is enough
to erase the kernel win and make the swap a net loss (0.93x end to end).

So this adapter normalizes ``q``/``k`` with FLA's Triton ``l2norm`` and passes
``use_qk_l2norm_in_kernel=False``, which is the configuration that measures
**2.54x forward / 2.41x end to end** over the ``musa`` backend. Nothing else about
the call site changes.

The adapter is deliberately signature-compatible with
``fla.ops.gated_delta_rule.chunk_gated_delta_rule`` so the generated
``Qwen3_5GatedDeltaNet.forward`` needs no branch, and it returns the same
``(output, last_recurrent_state)`` pair.

Supported (the training path this backend exists for):

- packed varlen, ``cu_seqlens`` given, ``B == 1`` -- which is how the collator
  packs the batch;
- ``initial_state=None``, ``output_final_state=False``;
- bf16 ``q``/``k``/``v``/``beta``, fp32 ``g``, head dim 64 or 128;
- every packed segment spanning **at least two 64-token chunks** (128 tokens) --
  ``torch_kernels`` raises ``ValueError`` below that, since its chunk maps assume
  a whole chunk per segment.

Not supported, and rejected up front rather than left to produce wrong numbers:

- dense calls (``cu_seqlens=None``): the TileLang backward needs per-segment chunk
  maps, and the padded state handoffs are not what the FLA backend returns;
- ``output_final_state=True``: with ``cu_seqlens`` the TileLang backend raises
  ``NotImplementedError`` (no per-sequence state for a packed batch), and without
  it, it computes the state over a sequence padded to a 64-token boundary --
  measured at 0.24-0.30 rms_rel against the FLA backend whenever ``S % 64 != 0``,
  i.e. a silently wrong cache. Inference with ``use_cache`` must stay on ``musa``
  until ``torch_kernels`` grows a per-sequence state path;
- ``initial_state``: same padded-state problem, so it is refused as well.

Hardware: MUSA only (the TileLang kernels are compiled with ``target="musa"``).
"""

from __future__ import annotations

import torch


def _require_torch_kernels():
    """Import the TileLang GDN entry point, with an actionable error if absent."""
    try:
        from torch_kernels.attention import gated_delta_net
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "The 'musa_tilelang' chunk_gated_delta_rule backend requires the 'torch_kernels' "
            "package (MUSA TileLang GDN kernels). Install it, or select "
            "chunk_gated_delta_rule_implementation='musa'/'fla' instead."
        ) from exc
    return gated_delta_net


def _reject_unsupported_paths(initial_state, output_final_state, cu_seqlens) -> None:
    """Fail loudly on the paths ``torch_kernels`` cannot serve correctly.

    Every one of these either raises deep inside ``torch_kernels`` or -- worse --
    returns a wrong number. They are checked here so the message names the backend
    and the alternative, instead of surfacing as a stack trace mid-training.
    """
    if cu_seqlens is None:
        raise NotImplementedError(
            "the 'musa_tilelang' chunk_gated_delta_rule backend only supports packed varlen "
            "input (cu_seqlens is required): its backward needs per-segment chunk maps. "
            "Select chunk_gated_delta_rule_implementation='musa' or 'fla' for dense batches."
        )
    if output_final_state:
        raise NotImplementedError(
            "the 'musa_tilelang' chunk_gated_delta_rule backend does not support "
            "output_final_state=True: torch_kernels has no per-sequence state for a packed "
            "batch, and its dense path computes the state over a sequence padded to a "
            "64-token boundary (measured 0.24-0.30 rms_rel against the 'musa' backend when "
            "S % 64 != 0). Keep inference / use_cache on chunk_gated_delta_rule_implementation="
            "'musa' or 'fla'."
        )
    if initial_state is not None:
        raise NotImplementedError(
            "the 'musa_tilelang' chunk_gated_delta_rule backend does not support "
            "initial_state: it shares the padded-state handoff that makes output_final_state "
            "unsafe here. Use chunk_gated_delta_rule_implementation='musa' or 'fla'."
        )


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    **kwargs,
):
    """Chunk gated delta rule on MUSA via ``torch_kernels`` (TileLang).

    ``q``/``k``/``v`` are ``[B, T, H, D]`` bf16, ``g`` is fp32 ``[B, T, H]`` and
    ``beta`` bf16 ``[B, T, H]`` -- the same contract the ``fla`` backend takes.

    When ``use_qk_l2norm_in_kernel`` is set, the normalization is applied here with
    FLA's Triton ``l2norm`` and the TileLang call is told the inputs are already
    normalized. That is numerically equivalent to what FLA does in-kernel and ~4x
    cheaper than letting ``torch_kernels`` run its own torch-level normalizer.

    Unsupported argument combinations raise ``NotImplementedError`` rather than
    return a wrong result; see the module docstring for the list.
    """
    gated_delta_net = _require_torch_kernels()

    if kwargs:
        # Deliberately stricter than the fla backend, which swallows **kwargs. The
        # generated forward passes none today; if it starts passing one of FLA's extras
        # (cu_seqlens_cpu, chunk_indices, cp_context, ...) this raises and names it,
        # rather than silently dropping something that changes the result.
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"musa_tilelang chunk_gated_delta_rule got unsupported kwargs: {unexpected}")

    _reject_unsupported_paths(initial_state, output_final_state, cu_seqlens)

    if use_qk_l2norm_in_kernel:
        from fla.modules.l2norm import l2norm

        q = l2norm(q)
        k = l2norm(k)

    out, final_state = gated_delta_net(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=cu_seqlens,
        backend="tilelang",
    )
    return out, final_state


__all__ = ["chunk_gated_delta_rule"]
