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

"""Numerical contract for the ``musa_tilelang`` chunk gated delta rule backend.

The backend must be a drop-in for the ``musa`` (tuned FLA) backend the Qwen3.5
GatedDeltaNet uses today: same call signature, same ``(output, final_state)``
pair, gradients in the same place.

Two things this file is deliberately explicit about, because getting either wrong
produces large, misleading errors:

* **q/k reach the kernel pre-normalized.** ``musa_tilelang`` applies FLA's
  ``l2norm`` itself when ``use_qk_l2norm_in_kernel=True``. Every test here either
  passes raw q/k with the flag on, or already-normalized q/k with the flag off --
  never both, which would normalize twice.
* **Input magnitudes matter.** The model feeds hidden states of O(1e-1), not
  ``N(0, 1)``; a ``g`` is built the way the module builds it
  (``-exp(A_log) * softplus(a + dt_bias)``) so it stays a bounded log-decay.
"""

import pytest
import torch

import veomni.ops  # noqa: F401 - trigger kernel registrations
from veomni.ops.kernel_registry import KERNEL_REGISTRY


pytestmark = pytest.mark.skipif(
    not hasattr(torch, "musa") or not torch.musa.is_available(),
    reason="the musa_tilelang GDN backend is MUSA-only",
)

HEADS = 32
DIM = 128
CHUNK = 64
KEYS = ("q", "k", "v", "g", "beta")
SCALE = DIM**-0.5


def _backend(name):
    return KERNEL_REGISTRY.resolve("chunk_gated_delta_rule", "standard", name)


def _skip_if_unavailable():
    try:
        _backend("musa_tilelang")
    except (KeyError, RuntimeError, ImportError) as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"musa_tilelang backend unavailable: {exc}")


def _chunk_lengths(seq_len, num_docs):
    base, rem = divmod(seq_len // CHUNK, num_docs)
    return [(base + 1 if i < rem else base) * CHUNK for i in range(num_docs)]


def _inputs(seq_len, num_docs, seed=0):
    """RAW (unnormalized) ``q``/``k`` -- exactly what the model hands the kernel."""
    device = torch.device("musa")
    torch.manual_seed(seed)
    lengths = _chunk_lengths(seq_len, num_docs)
    offsets, acc = [0], 0
    for length in lengths:
        acc += length
        offsets.append(acc)
    cu_seqlens = torch.tensor(offsets, device=device, dtype=torch.int32)
    shape = (1, seq_len, HEADS, DIM)

    def mk():
        return torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.1

    a_log = torch.rand(HEADS, device=device, dtype=torch.float32) * 4.0
    dt_bias = torch.rand(HEADS, device=device, dtype=torch.float32)
    a_proj = torch.randn(1, seq_len, HEADS, device=device, dtype=torch.float32) * 0.1
    g = (-a_log.exp() * torch.nn.functional.softplus(a_proj + dt_bias)).contiguous()
    beta = torch.sigmoid(torch.randn(1, seq_len, HEADS, device=device, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    return device, cu_seqlens, [mk(), mk(), mk(), g, beta], lengths


def _stats(got, ref):
    diff = (got.float() - ref.float()).abs()
    return diff.max().item(), (diff.pow(2).mean().sqrt() / ref.float().pow(2).mean().sqrt()).item()


def _forward(kernel, tensors, cu_seqlens, l2):
    q, k, v, g, beta = (t.detach().clone() for t in tensors)
    return kernel(
        q,
        k,
        v,
        g=g,
        beta=beta,
        scale=SCALE,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=l2,
        cu_seqlens=cu_seqlens,
    )


def _raw_qk_grads(kernel, tensors, cu_seqlens):
    """Forward the kernel (``l2=False``) and backprop to the RAW q/k.

    Normalization is applied with FLA's autograd-capable ``l2norm`` and lives
    *outside* the kernel call, so ``q.grad`` is dL/d(raw q) -- the same quantity a
    reference that normalizes its own inputs produces. Passing
    ``use_qk_l2norm_in_kernel=True`` here would normalize a second time.
    """
    from fla.modules.l2norm import l2norm

    q, k, v, g, beta = (t.detach().clone().requires_grad_(True) for t in tensors)
    out, _ = kernel(
        l2norm(q),
        l2norm(k),
        v,
        g=g,
        beta=beta,
        scale=SCALE,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=cu_seqlens,
    )
    out.float().sum().backward()
    return out.detach(), {n: t.grad.detach() for n, t in zip(KEYS, (q, k, v, g, beta))}


def test_backend_is_registered():
    assert "musa_tilelang" in KERNEL_REGISTRY.list_available("chunk_gated_delta_rule", "standard")


@pytest.mark.parametrize("seq_len,num_docs", [(4096, 1), (4096, 8), (8512, 17)])
def test_matches_per_document_fp32_reference(seq_len, num_docs):
    """Varlen semantics: each document is an independent sequence.

    ``transformers.torch_chunk_gated_delta_rule`` has no ``cu_seqlens`` argument, so
    the only fair fp32 reference for a packed input is one eager call per document,
    each normalizing its own inputs.
    """
    _skip_if_unavailable()
    from fla.modules.l2norm import l2norm
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import torch_chunk_gated_delta_rule

    _, cu_seqlens, tensors, lengths = _inputs(seq_len, num_docs, seed=11)
    outs, offset = [], 0
    grads = {n: torch.zeros_like(t) for n, t in zip(KEYS, tensors)}
    for length in lengths:
        pieces = [t[:, offset : offset + length].detach().clone().requires_grad_(True) for t in tensors]
        out, _ = torch_chunk_gated_delta_rule(
            l2norm(pieces[0]),
            l2norm(pieces[1]),
            pieces[2],
            g=pieces[3],
            beta=pieces[4],
            scale=SCALE,
            use_qk_l2norm_in_kernel=False,
        )
        out.float().sum().backward()
        outs.append(out.detach())
        for name, piece in zip(KEYS, pieces):
            grads[name][:, offset : offset + length] = piece.grad.detach()
        offset += length
    ref_out = torch.cat(outs, dim=1)

    out, got = _raw_qk_grads(_backend("musa_tilelang"), tensors, cu_seqlens)
    max_abs, rms_rel = _stats(out, ref_out)
    assert rms_rel < 2e-2, f"output rms_rel={rms_rel:.3e} max_abs={max_abs:.3e}"
    for name in KEYS:
        _, rms_rel_g = _stats(got[name], grads[name])
        assert rms_rel_g < 3e-2, f"d{name} rms_rel={rms_rel_g:.3e}"


@pytest.mark.parametrize("seq_len,num_docs", [(4096, 1), (4096, 8), (8512, 17)])
def test_matches_tuned_fla_backend(seq_len, num_docs):
    """``musa_tilelang`` and ``musa`` must agree to bf16 noise, forward and backward."""
    _skip_if_unavailable()
    _, cu_seqlens, tensors, _ = _inputs(seq_len, num_docs, seed=5)
    fla_out, fla_grads = _raw_qk_grads(_backend("musa"), tensors, cu_seqlens)
    tk_out, tk_grads = _raw_qk_grads(_backend("musa_tilelang"), tensors, cu_seqlens)
    max_abs, rms_rel = _stats(tk_out, fla_out)
    assert rms_rel < 1e-2, f"output rms_rel={rms_rel:.3e} max_abs={max_abs:.3e}"
    for name in KEYS:
        _, rms_rel_g = _stats(tk_grads[name], fla_grads[name])
        assert rms_rel_g < 2e-2, f"d{name} rms_rel={rms_rel_g:.3e}"


def test_l2_flag_path_matches_the_fp32_reference(seq_len=4096, num_docs=8):
    """The flag-ON path (raw q/k in, normalization inside) must match the fp32 reference.

    The other numerical tests feed pre-normalized q/k with the flag off, so they never
    exercise the adapter's own normalization. This one drives the path the model
    actually calls -- raw q/k, ``use_qk_l2norm_in_kernel=True`` -- and checks it
    against ``transformers``' eager implementation, which normalizes for itself.
    A wrong axis, a wrong eps, or normalizing ``v`` by mistake only shows up here.
    """
    _skip_if_unavailable()
    from fla.modules.l2norm import l2norm
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import torch_chunk_gated_delta_rule

    _, cu_seqlens, tensors, lengths = _inputs(seq_len, num_docs, seed=13)
    out, _ = _forward(_backend("musa_tilelang"), tensors, cu_seqlens, l2=True)

    outs, offset = [], 0
    for length in lengths:
        pieces = [t[:, offset : offset + length].detach() for t in tensors]
        ref, _ = torch_chunk_gated_delta_rule(
            l2norm(pieces[0]),
            l2norm(pieces[1]),
            pieces[2],
            g=pieces[3],
            beta=pieces[4],
            scale=SCALE,
            use_qk_l2norm_in_kernel=False,
        )
        outs.append(ref)
        offset += length
    _, rms_rel = _stats(out, torch.cat(outs, dim=1))
    assert rms_rel < 2e-2, f"flag-on path rms_rel={rms_rel:.3e}"


def test_unsupported_paths_raise(seq_len=4096, num_docs=2):
    """Every path the TileLang backend cannot serve must raise, not return rubbish.

    Dense input and the two state handoffs: the first because the backward needs
    per-segment chunk maps, the others because ``torch_kernels`` computes the state
    over a 64-token-padded sequence (0.24-0.30 rms_rel against ``musa`` when
    ``S % 64 != 0``).
    """
    _skip_if_unavailable()
    kernel = _backend("musa_tilelang")
    device, cu_seqlens, tensors, _ = _inputs(seq_len, num_docs)
    q, k, v, g, beta = tensors
    state = torch.zeros(1, HEADS, DIM, DIM, device=device, dtype=torch.float32)

    def call(**overrides):
        kwargs = dict(
            scale=SCALE,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
        kwargs.update(overrides)
        return kernel(q, k, v, g=g, beta=beta, **kwargs)

    with pytest.raises(NotImplementedError, match="packed varlen"):
        call(cu_seqlens=None)
    with pytest.raises(NotImplementedError, match="output_final_state"):
        call(output_final_state=True)
    with pytest.raises(NotImplementedError, match="initial_state"):
        call(initial_state=state)


def test_l2_flag_matches_pre_normalized_inputs(seq_len=4096, num_docs=1):
    """``use_qk_l2norm_in_kernel=True`` must equal passing pre-normalized q/k.

    The backend moves the normalization out of the TileLang call, so this is the
    invariant that keeps it equivalent to FLA's in-kernel normalization. Not
    bitwise: the two schedules differ.
    """
    _skip_if_unavailable()
    from fla.modules.l2norm import l2norm

    _, cu_seqlens, tensors, _ = _inputs(seq_len, num_docs, seed=7)
    kernel = _backend("musa_tilelang")
    with_norm, _ = _forward(kernel, tensors, cu_seqlens, l2=True)
    q, k, v, g, beta = tensors
    pre_norm, _ = _forward(kernel, [l2norm(q.clone()), l2norm(k.clone()), v, g, beta], cu_seqlens, l2=False)
    _, rms_rel = _stats(with_norm, pre_norm)
    assert rms_rel < 1e-6, f"flag and pre-normalized paths disagree: rms_rel={rms_rel:.3e}"


def test_l2_flag_actually_changes_the_result(seq_len=4096, num_docs=1):
    """The flag must not be silently ignored."""
    _skip_if_unavailable()
    _, cu_seqlens, tensors, _ = _inputs(seq_len, num_docs, seed=9)
    kernel = _backend("musa_tilelang")
    off, _ = _forward(kernel, tensors, cu_seqlens, l2=False)
    on, _ = _forward(kernel, tensors, cu_seqlens, l2=True)
    _, rms_rel = _stats(on, off)
    assert rms_rel > 1e-3, "flag had no effect: normalization was not applied"


def test_rejects_unknown_kwargs():
    _skip_if_unavailable()
    _, cu_seqlens, tensors, _ = _inputs(4096, 1)
    q, k, v, g, beta = tensors
    with pytest.raises(TypeError, match="unsupported kwargs"):
        _backend("musa_tilelang")(
            q,
            k,
            v,
            g=g,
            beta=beta,
            scale=SCALE,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
            not_a_real_kwarg=1,
        )
