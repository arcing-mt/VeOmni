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

import functools
from contextlib import contextmanager
from contextvars import ContextVar

import torch


def _probe_accepts_host_cu_seqlens(gdn_chunk) -> bool:
    """Whether one ``torch_kernels`` GDN entry point takes ``cu_seqlens_cpu``.

    Duck-typed off the code object: a callable the probe cannot introspect (a
    ``partial``, a builtin) reports ``False``, which costs the optimisation but
    never changes a result.
    """
    code = getattr(gdn_chunk, "__code__", None)
    return code is not None and "cu_seqlens_cpu" in code.co_varnames[: code.co_argcount + code.co_kwonlyargcount]


@functools.lru_cache(maxsize=1)
def _accepts_host_cu_seqlens() -> bool:
    """Whether the installed ``torch_kernels`` takes host-side packed offsets.

    ``torch_kernels`` is not a declared dependency of VeOmni -- the MUSA
    environment provides it -- so an install that predates ``cu_seqlens_cpu`` has
    to keep working. It does: the offsets are simply not handed over, and the
    backend pays its device-to-host sync exactly as it did before.
    """
    try:
        from torch_kernels.attention.tilelang.flash_linear_attention.gdn_chunk_bwd import gdn_chunk
    except ImportError:  # a layout this probe does not know: assume the older ABI
        return False
    return _probe_accepts_host_cu_seqlens(gdn_chunk)


#: Host-side packed offsets for the GDN forward running on this thread.
#:
#: The generated ``GatedDeltaNet.forward`` builds its kernel call itself, so a
#: caller cannot hand these in as an argument without regenerating the model; the
#: MUSA runtime patch scopes them around the decoder layer instead (see
#: ``qwen3_5_musa_runtime_patch.install_qwen3_5_gdn_cpu_seqlens_patch``), which is
#: also the block gradient checkpointing re-runs. A context variable is what makes
#: that safe: the recompute runs on the autograd thread, and the value stays bound
#: to the forward that set it rather than to a module attribute two threads would
#: race on. Defaults to ``None`` -- without the patch, nothing changes.
_forward_cu_seqlens_cpu: ContextVar[tuple[int, ...] | None] = ContextVar("gdn_forward_cu_seqlens_cpu", default=None)


@contextmanager
def forward_cu_seqlens_cpu(cu_seqlens_cpu):
    """Scope host-side packed offsets to every GDN forward nested in this block."""
    token = _forward_cu_seqlens_cpu.set(tuple(cu_seqlens_cpu))
    try:
        yield
    finally:
        _forward_cu_seqlens_cpu.reset(token)


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
    *,
    cu_seqlens_cpu: list[int] | None = None,
    **kwargs,
):
    """Chunk gated delta rule on MUSA via ``torch_kernels`` (TileLang).

    ``q``/``k``/``v`` are ``[B, T, H, D]`` bf16, ``g`` is fp32 ``[B, T, H]`` and
    ``beta`` bf16 ``[B, T, H]`` -- the same contract the ``fla`` backend takes.

    When ``use_qk_l2norm_in_kernel`` is set, the normalization is applied here with
    FLA's Triton ``l2norm`` and the TileLang call is told the inputs are already
    normalized. That is numerically equivalent to what FLA does in-kernel and ~4x
    cheaper than letting ``torch_kernels`` run its own torch-level normalizer.

    ``cu_seqlens_cpu`` is the host twin of ``cu_seqlens`` -- the same ``[NS + 1]``
    cumulative offsets, as a plain sequence. Passing it is worth more than it looks:
    ``torch_kernels`` learns the per-sequence lengths by reading the device vector
    (``varlen_lengths``), and that read is a blocking device-to-host sync which
    drains the queue. It is reached once per layer per forward and once more per
    layer by the gradient-checkpoint recompute: 60 syncs and ~430 ms of CPU per step
    on the 30-layer Qwen3.5-35B-A3B MUSA run, 12.9 ms of it per recompute call.
    Callers that hold the offsets on the host (the collator does) pass them here or
    scope them with ``forward_cu_seqlens_cpu``; otherwise the backend syncs.

    ``cu_seqlens_cpu`` is keyword-only on purpose: it keeps the positional layout of
    the FLA entry point this adapter mirrors, so a caller that passes FLA's
    ``head_first`` positionally cannot silently bind it here.

    Unsupported argument combinations raise ``NotImplementedError`` rather than
    return a wrong result; see the module docstring for the list.
    """
    gated_delta_net = _require_torch_kernels()

    if kwargs:
        # Deliberately stricter than the fla backend, which swallows **kwargs. The
        # generated forward passes none today; if it starts passing one of FLA's extras
        # (chunk_indices, cp_context, ...) this raises and names it, rather than
        # silently dropping something that changes the result.
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"musa_tilelang chunk_gated_delta_rule got unsupported kwargs: {unexpected}")

    _reject_unsupported_paths(initial_state, output_final_state, cu_seqlens)

    if cu_seqlens_cpu is None:
        # Scoped by the MUSA runtime patch around the decoder layer that is about to
        # call us; ``None`` when the patch is not installed or the batch carries no
        # host offsets, in which case the backend falls back to the device read.
        cu_seqlens_cpu = _forward_cu_seqlens_cpu.get()

    if cu_seqlens_cpu is not None and cu_seqlens is not None and len(cu_seqlens_cpu) != cu_seqlens.numel():
        # The host offsets ARE the segmentation, and torch_kernels only checks that
        # they sum to the packed length -- a same-total pair with different boundaries
        # would resegment silently. Comparing entry counts is sync-free and catches
        # the realistic failure, which is one of the two being re-derived downstream.
        raise ValueError(
            f"cu_seqlens_cpu has {len(cu_seqlens_cpu)} entries but cu_seqlens has "
            f"{cu_seqlens.numel()}; both must describe the same segmentation"
        )

    if use_qk_l2norm_in_kernel:
        from fla.modules.l2norm import l2norm

        q = l2norm(q)
        k = l2norm(k)

    # Only hand the offsets over when the installed torch_kernels can take them; a
    # pre-``cu_seqlens_cpu`` build would otherwise reject the keyword outright.
    host_offsets = (
        {"cu_seqlens_cpu": cu_seqlens_cpu} if cu_seqlens_cpu is not None and _accepts_host_cu_seqlens() else {}
    )

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
        **host_offsets,
    )
    return out, final_state


__all__ = ["chunk_gated_delta_rule", "forward_cu_seqlens_cpu"]
