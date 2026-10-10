"""Numerical and fallback contracts for opt-in Qwen3.5 MUSA compute paths."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from veomni.ops.kernels.cross_entropy import chunk_loss as chunk
from veomni.ops.kernels.cross_entropy import supervised_chunk_loss as supervised
from veomni.utils.device import IS_MUSA_AVAILABLE


@pytest.mark.parametrize("shifted", [False, True])
@pytest.mark.parametrize("mask", ["partial", "none", "all"])
def test_supervised_ce_matches_full_projection(monkeypatch, shifted, mask):
    monkeypatch.setattr(chunk, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    monkeypatch.setattr(supervised, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    torch.manual_seed(8)
    hidden = torch.randn(2, 7, 5, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(11, 5, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, 11, (2, 7))
    if mask == "partial":
        labels[:, ::2] = -100
    elif mask == "all":
        labels.fill_(-100)
    targets = labels if shifted else labels[:, 1:]
    features = hidden if shifted else hidden[:, :-1]
    expected = F.cross_entropy(F.linear(features, weight).float().reshape(-1, 11), targets.reshape(-1))
    actual, logits = supervised.compact_chunk_loss(
        hidden, weight, labels, chunk_size=3, shift_labels=labels if shifted else None
    )
    assert logits is None
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6, equal_nan=True)
    reference_grads = torch.autograd.grad(expected * 0.375, (hidden, weight), retain_graph=True)
    candidate_grads = torch.autograd.grad(actual * 0.375, (hidden, weight))
    for got, want in zip(candidate_grads, reference_grads):
        torch.testing.assert_close(got, want, rtol=2e-6, atol=2e-7)


def test_supervised_ce_sp_preserves_original(monkeypatch):
    monkeypatch.setattr(supervised, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=True))
    sentinel = object()
    monkeypatch.setattr(supervised, "original", lambda *a, **k: sentinel)
    assert supervised.compact_chunk_loss(torch.empty(1, 3, 2), torch.empty(4, 2), torch.zeros(1, 3)) is sentinel


def test_cpu_opt_in_retains_native_loss(monkeypatch):
    monkeypatch.setenv("VEOMNI_MUSA_SUPERVISED_CE", "1")
    sentinel = object()
    monkeypatch.setattr(chunk, "_native_chunk_loss_function", lambda *a, **k: sentinel)
    assert chunk.chunk_loss_function(torch.empty(1, 3, 2)) is sentinel


def test_vision_embedding_cpu_fallback():
    pytest.importorskip("triton")
    from veomni.models.transformers.qwen3_5_moe.vision_embedding_csr import cached_embedding

    embedding = torch.nn.Embedding(7, 3)
    indices = torch.tensor([[1, 3, 1]])
    expected = embedding(indices)
    actual = cached_embedding(embedding, indices, (1, 3, 48))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), embedding.weight)[0],
        torch.autograd.grad(expected.sum(), embedding.weight)[0],
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not IS_MUSA_AVAILABLE, reason="Qualified MUSA/FLA build required")
def test_qk_preentry_preserves_native_forward_backward():
    from veomni.models.transformers.qwen3_5_moe import qk_repeat_l2norm as qk

    device = torch.device("musa", torch.musa.current_device())
    torch.manual_seed(3)
    storage = torch.randn(1, 257, 64, 128, dtype=torch.bfloat16, device=device, requires_grad=True)
    x = storage[:, :, :16]
    native = qk.native(x)
    candidate = qk.repeat_norm(x)
    dy = torch.randn_like(native)
    torch.testing.assert_close(candidate, native, rtol=0, atol=0)
    expected = torch.autograd.grad(native, x, dy, retain_graph=True)[0]
    actual = torch.autograd.grad(candidate, x, dy)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert qk.stats["fused_forward"] > 0


@pytest.mark.skipif(not IS_MUSA_AVAILABLE, reason="Qualified MUSA embedding backward required")
def test_vision_csr_matches_native_after_cache_and_stream_change():
    from veomni.models.transformers.qwen3_5_moe import vision_embedding_csr as csr

    device = torch.device("musa", torch.musa.current_device())
    embedding = torch.nn.Embedding(2304, 1152, dtype=torch.bfloat16, device=device)
    h, w, grid = 48, 32, 48
    hi = torch.linspace(0, grid - 1, h, dtype=torch.float64, device=device).long()
    wi = torch.linspace(0, grid - 1, w, dtype=torch.float64, device=device).long()
    hf, wf = torch.meshgrid(hi, wi, indexing="ij")
    hc, wc = torch.meshgrid((hi + 1).clamp_max(grid - 1), (wi + 1).clamp_max(grid - 1), indexing="ij")
    indices = (torch.stack([hf, hf, hc, hc]) * grid + torch.stack([wf, wc, wf, wc])).reshape(4, -1)
    expected = embedding(indices)
    dy = torch.randn_like(expected)
    expected_dw = torch.autograd.grad(expected, embedding.weight, dy)[0]
    before = dict(csr.counters)
    first = csr.cached_embedding(embedding, indices, (h, w, grid))
    first_dw = torch.autograd.grad(first, embedding.weight, dy)[0]
    side = torch.musa.Stream(device=device)
    side.wait_stream(torch.musa.current_stream(device))
    with torch.musa.stream(side):
        second = csr.cached_embedding(embedding, indices, (h, w, grid))
        second_dw = torch.autograd.grad(second, embedding.weight, dy)[0]
    side.synchronize()
    for got in [first, second]:
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
    for got in [first_dw, second_dw]:
        torch.testing.assert_close(got, expected_dw, rtol=0, atol=0)
    assert csr.counters["cache_misses"] - before.get("cache_misses", 0) == 1
    assert csr.counters["cache_hits"] - before.get("cache_hits", 0) == 1
    native_loss = embedding(indices).square().sum()
    candidate_loss = csr.cached_embedding(embedding, indices, (h, w, grid)).square().sum()
    native_first = torch.autograd.grad(native_loss, embedding.weight, create_graph=True)[0]
    candidate_first = torch.autograd.grad(candidate_loss, embedding.weight, create_graph=True)[0]
    torch.testing.assert_close(candidate_first, native_first, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(candidate_first.sum(), embedding.weight)[0],
        torch.autograd.grad(native_first.sum(), embedding.weight)[0],
        rtol=0,
        atol=0,
    )
    assert csr.counters["higher_order_native_backward"] > 0


@pytest.mark.skipif(not IS_MUSA_AVAILABLE, reason="Qualified MUSA/FLA/torch_kernels GDN required")
def test_installed_qk_adapter_matches_original_gdn(monkeypatch):
    pytest.importorskip("torch_kernels.attention", reason="Optional private torch_kernels overlay required")
    from fla.modules.convolution import causal_conv1d
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

    from veomni.models.transformers.qwen3_5_moe import musa_qk_preentry as adapter
    from veomni.models.transformers.qwen3_5_moe.generated import patched_modeling_qwen3_5_moe_gpu as modeling
    from veomni.ops.kernels.gated_delta_rule.musa_tilelang import chunk_gated_delta_rule

    monkeypatch.setattr(modeling, "get_parallel_state", lambda: SimpleNamespace(ulysses_enabled=False))
    monkeypatch.setattr(modeling, "veomni_rms_norm_gated", SimpleNamespace(use_non_eager_impl=False))
    monkeypatch.setattr(modeling, "veomni_causal_conv1d", SimpleNamespace(bound_kernel=lambda: causal_conv1d))
    monkeypatch.setattr(
        modeling, "veomni_chunk_gated_delta_rule", SimpleNamespace(bound_kernel=lambda: chunk_gated_delta_rule)
    )
    # Restore both the class and installer ownership after the test.
    cls = modeling.Qwen3_5MoeGatedDeltaNet
    original = cls.forward
    monkeypatch.setattr(cls, "forward", original)
    for attr in ["_original", "_wrapper", "_backend", "_model_module"]:
        monkeypatch.setattr(adapter, attr, None)
    monkeypatch.setenv("VEOMNI_MUSA_QK_PREENTRY", "1")
    torch.manual_seed(12)
    config = Qwen3_5MoeTextConfig(
        hidden_size=2048,
        num_hidden_layers=1,
        layer_types=["linear_attention"],
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    device = torch.device("musa", torch.musa.current_device())
    layer = cls(config, 0).to(device=device, dtype=torch.bfloat16)
    hidden = torch.randn(1, 257, 2048, device=device, dtype=torch.bfloat16, requires_grad=True)
    lengths = torch.tensor([0, 257], device=device, dtype=torch.int32)
    expected = original(layer, hidden, cu_seq_lens_q=lengths)
    adapter.install(modeling)
    before = adapter.counts["preentry_fast"]
    actual = layer(hidden, cu_seq_lens_q=lengths)
    dy = torch.randn_like(expected)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    inputs = (hidden, *tuple(layer.parameters()))
    reference_grads = torch.autograd.grad(expected, inputs, dy, retain_graph=True)
    candidate_grads = torch.autograd.grad(actual, inputs, dy)
    for got, want in zip(candidate_grads, reference_grads):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert adapter.counts["preentry_fast"] > before
