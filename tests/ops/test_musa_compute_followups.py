"""Contracts for incremental optimizations over the guarded MUSA paths."""

from types import SimpleNamespace

import pytest
import torch

from veomni.utils.device import IS_MUSA_AVAILABLE


@pytest.mark.skipif(not IS_MUSA_AVAILABLE, reason="MUSA interpolation parity required")
def test_vision_metadata_reuse_stream_eviction_and_weight_updates(monkeypatch):
    from veomni.models.transformers.qwen3_5_moe import musa_vision_embedding as adapter
    from veomni.models.transformers.qwen3_5_moe import vision_interpolation_cache as cache
    from veomni.models.transformers.qwen3_5_moe.generated import patched_modeling_qwen3_5_moe_gpu as modeling

    class Vision(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pos_embed = torch.nn.Embedding(2304, 1152, device="musa", dtype=torch.bfloat16)
            self.num_grid_per_side = 48
            self.spatial_merge_size = 2

        @property
        def device(self):
            return self.pos_embed.weight.device

    reference = adapter.make_candidate(modeling.Qwen3_5MoeVisionModel.fast_pos_embed_interpolate)
    candidate = adapter.make_candidate(modeling.Qwen3_5MoeVisionModel.fast_pos_embed_interpolate, cache_metadata=True)
    torch.manual_seed(19)
    module = Vision()
    grids = [[1, 26, 40], [2, 26, 40], [1, 12, 12]]
    before = dict(cache.counters)
    expected = reference(module, grids)
    actual = candidate(module, grids)
    dy = torch.randn_like(expected)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual, module.pos_embed.weight, dy)[0],
        torch.autograd.grad(expected, module.pos_embed.weight, dy)[0],
        rtol=0,
        atol=0,
    )
    assert cache.counters["misses"] - before.get("misses", 0) == 2
    assert cache.counters["hits"] - before.get("hits", 0) == 1
    with torch.no_grad():
        module.pos_embed.weight.add_(0.25)
    expected = reference(module, grids)
    side = torch.musa.Stream()
    side.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(side):
        actual = candidate(module, grids)
        got_grad = torch.autograd.grad(actual, module.pos_embed.weight, dy)[0]
    side.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(got_grad, torch.autograd.grad(expected, module.pos_embed.weight, dy)[0], rtol=0, atol=0)
    # Saved coefficients must survive eviction before their backward executes.
    monkeypatch.setattr(cache, "MAX_CACHED_GRIDS", 1)
    evictions = cache.counters["evictions"]
    cache._cache.pop(module)
    small_grids = [[1, 10, 12]]
    expected = reference(module, small_grids)
    actual = candidate(module, small_grids)
    candidate(module, [[1, 14, 12]])
    assert cache.counters["evictions"] == evictions + 1
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), module.pos_embed.weight)[0],
        torch.autograd.grad(expected.sum(), module.pos_embed.weight)[0],
        rtol=0,
        atol=0,
    )
    # Inference-created cache entries must be ordinary tensors in later training.
    cache._cache.pop(module)
    with torch.inference_mode():
        candidate(module, small_grids)
    candidate(module, small_grids).sum().backward()
    assert not torch.is_inference(cache._cache[module][next(iter(cache._cache[module]))][0])


@pytest.mark.skipif(not IS_MUSA_AVAILABLE, reason="MUSA TE state reuse parity required")
def test_ace_forward_only_compact_and_row_map_reuse(monkeypatch):
    from veomni.distributed.moe import deepep_ace as ace
    from veomni.ops.kernels.moe import musa_te_slot_probability as slots

    torch.manual_seed(13)
    cpu_indices = (torch.arange(129)[:, None] + torch.arange(8)).remainder(32)
    cpu_indices[::3, -1] = -1
    counts = torch.bincount(cpu_indices[cpu_indices >= 0], minlength=32).tolist()
    indices = cpu_indices.to("musa")
    hidden = torch.randn(129, 2048, device="musa", dtype=torch.bfloat16, requires_grad=True)
    probabilities = torch.randn(129, 8, device="musa", dtype=torch.float32, requires_grad=True)
    monkeypatch.setenv("VEOMNI_MUSA_ACE_TE_SLOT_PROBABILITY", "1")
    monkeypatch.setenv("VEOMNI_MUSA_ACE_COMPACT_STATE_REUSE", "0")
    expected = ace._te_compact_permute(hidden, indices, probabilities, 32, counts)
    dy, dp = torch.randn_like(expected[0]), torch.randn_like(expected[1])
    reference_grads = torch.autograd.grad(expected[:2], (hidden, probabilities), (dy, dp))
    monkeypatch.setenv("VEOMNI_MUSA_ACE_COMPACT_STATE_REUSE", "1")
    before = slots.COUNTERS["forward_only_compact"]
    plan = {"_reuse_forward_row_nt": True}
    actual = ace._te_compact_permute(hidden, indices, probabilities, 32, counts, plan=plan)
    assert slots.COUNTERS["forward_only_compact"] == before
    torch.testing.assert_close(plan.pop("_forward_row_nt"), actual[4].t().contiguous(), rtol=0, atol=0)
    assert "_reuse_forward_row_nt" not in plan
    for got, want in zip(torch.autograd.grad(actual[:2], (hidden, probabilities), (dy, dp)), reference_grads):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    with torch.no_grad():
        untracked = ace._te_compact_permute(hidden, indices, probabilities, 32, counts)
    assert slots.COUNTERS["forward_only_compact"] == before + 1
    for got, want in zip(untracked[:2], expected[:2]):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("mask", ["partial", "all", "none"])
@pytest.mark.parametrize("device_type", ["cpu", "musa"])
def test_supervised_metadata_causal_alignment(monkeypatch, mask, device_type):
    from veomni.ops.kernels.cross_entropy import chunk_loss as chunk
    from veomni.ops.kernels.cross_entropy import supervised_chunk_loss as supervised
    from veomni.ops.kernels.cross_entropy import supervised_metadata as metadata

    monkeypatch.setattr(chunk, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    monkeypatch.setattr(supervised, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    if device_type == "musa" and not IS_MUSA_AVAILABLE:
        pytest.skip("MUSA BF16 metadata path requires a device")
    torch.manual_seed(9)
    dtype = torch.bfloat16 if device_type == "musa" else torch.float32
    hidden = torch.randn(2, 9, 7, device=device_type, dtype=dtype, requires_grad=True)
    weight = torch.randn(13, 7, device=device_type, dtype=dtype, requires_grad=True)
    cpu_labels = torch.randint(0, 13, (2, 9))
    if mask == "partial":
        cpu_labels[:, 3::2] = -100
    elif mask == "all":
        cpu_labels.fill_(-100)
    labels = cpu_labels.to(device_type)
    monkeypatch.setenv("VEOMNI_MUSA_SUPERVISED_CE_METADATA", "0")
    expected, _ = supervised.compact_chunk_loss(hidden, weight, labels, chunk_size=3)
    expected_grad = torch.autograd.grad(expected, (hidden, weight))
    batch = {"labels": cpu_labels}
    metadata.prepare_metadata(batch)
    entry = batch["multimodal_metadata"][metadata.KEY]
    for key in ("indices", "targets"):
        entry[key] = entry[key].to(device_type)
    monkeypatch.setenv("VEOMNI_MUSA_SUPERVISED_CE_METADATA", "1")

    def reject_nonzero(*args, **kwargs):
        raise AssertionError("Metadata-backed loss must not execute dynamic nonzero")

    monkeypatch.setattr(torch, "nonzero", reject_nonzero)
    actual, _ = supervised.compact_chunk_loss(
        hidden, weight, labels, chunk_size=3, multimodal_metadata=batch["multimodal_metadata"]
    )
    loss_tolerance = 1e-6 if device_type == "musa" else 0
    torch.testing.assert_close(actual, expected, rtol=loss_tolerance, atol=loss_tolerance, equal_nan=True)
    for got, want in zip(torch.autograd.grad(actual, (hidden, weight)), expected_grad):
        torch.testing.assert_close(
            got, want, rtol=0.02 if device_type == "musa" else 0, atol=5e-4 if device_type == "musa" else 0
        )
    cached = batch["multimodal_metadata"]
    assert metadata.selection(labels, hidden, labels, -100, cached) is None
    cached[metadata.KEY]["label_shape"] = (1, 18)
    assert metadata.selection(labels, hidden, None, -100, cached) is None


@pytest.mark.skipif(not IS_MUSA_AVAILABLE, reason="MUSA integer routing counts required")
@pytest.mark.parametrize("tokens", [1, 257, 8192, 24577, 28609, 32705, 45001, 65536])
def test_fused_counts_match_cpu_including_old_shape_gaps(tokens):
    from veomni.ops.kernels.moe.musa_te_count_kernel import fused_column_counts

    cpu_mask = (torch.arange(tokens)[:, None] + torch.arange(32)[None, :]).remainder(11) < 4
    expected = cpu_mask.sum(0, dtype=torch.int64)
    actual = fused_column_counts(cpu_mask.to("musa"))
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    assert actual.dtype == torch.int64
    assert fused_column_counts(cpu_mask) is None


@pytest.mark.skipif(not IS_MUSA_AVAILABLE, reason="MUSA joint QK parity required")
@pytest.mark.parametrize("length", [257, 8192])
def test_joint_qk_matches_existing_preentry_and_native_backward(length):
    from veomni.models.transformers.qwen3_5_moe import qk_repeat_l2norm as qk

    torch.manual_seed(33)
    storage = torch.randn(1, length, 64, 128, device="musa", dtype=torch.bfloat16, requires_grad=True)
    query, key = storage[:, :, :16], storage[:, :, 16:32]
    # Populate the original forward autotune key before either candidate.
    qk.native(query)
    expected = (qk.repeat_norm(query), qk.repeat_norm(key))
    dy = tuple(torch.randn_like(x) for x in expected)
    expected_grad = torch.autograd.grad(expected, storage, dy)[0]
    before = qk.stats["joint_forward"]
    actual = qk.repeat_norm_pair(query, key)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual, storage, dy)[0], expected_grad, rtol=0, atol=0)
    assert qk.stats["joint_forward"] == before + 1


@pytest.mark.skipif(not IS_MUSA_AVAILABLE, reason="MUSA QK backward parity required")
@pytest.mark.parametrize("length,scale", [(257, 1.0), (8192, 1.0), (19, 1e-30), (19, 0.0)])
def test_qk_pair_backward_with_bf16_tolerance(monkeypatch, length, scale):
    from veomni.models.transformers.qwen3_5_moe import qk_repeat_l2norm as qk

    torch.manual_seed(27)
    storage = torch.randn(1, length, 64, 128, device="musa", dtype=torch.bfloat16, requires_grad=True)
    x = storage[:, :, :16]
    expected = qk.native(x)
    dy = torch.randn_like(expected) * scale
    reference_grad = torch.autograd.grad(expected, x, dy)[0]
    monkeypatch.setenv("VEOMNI_MUSA_QK_FUSED_BACKWARD", "1")
    count = qk.stats["fused_pair_backward"]
    actual = qk.repeat_norm(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual, x, dy)[0],
        reference_grad,
        rtol=0.02,
        atol=5e-4 * scale,
    )
    assert qk.stats["fused_pair_backward"] == count + 1


@pytest.mark.parametrize("mode", ["on", "off_ce", "off_metadata", "sp", "classification", "no_hook"])
def test_packing_collator_supervised_metadata_contract(monkeypatch, mode):
    from veomni.data import data_collator as collators
    from veomni.ops.kernels.cross_entropy.supervised_metadata import KEY
    from veomni.trainer.base import BaseTrainer

    monkeypatch.setattr(collators, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=mode == "sp"))
    monkeypatch.setenv("VEOMNI_MUSA_SUPERVISED_CE", "0" if mode == "off_ce" else "1")
    monkeypatch.setenv("VEOMNI_MUSA_SUPERVISED_CE_METADATA", "0" if mode == "off_metadata" else "1")

    def hook(batch, padding):
        batch["multimodal_metadata"] = {"existing_metadata": 17}

    collator = collators.PackingCollator(
        pad_to_length=8,
        seq_classification=mode == "classification",
        metadata_collate_func=hook if mode != "no_hook" else None,
    )
    features = [
        dict(input_ids=torch.tensor([1, 2, 3]), labels=torch.tensor([7, 8, -100]), position_ids=torch.arange(3)),
        dict(input_ids=torch.tensor([4, 5, 6]), labels=torch.tensor([9, 10, 11]), position_ids=torch.arange(3)),
    ]
    batch = collator(features)
    md = batch.get("multimodal_metadata", {})
    assert (KEY in md) == (mode == "on")
    if mode not in ("sp", "no_hook"):
        assert md["existing_metadata"] == 17
    if mode == "on":
        assert batch["labels"].tolist() == [[7, 8, -100, -100, 10, 11, -100, -100]]
        assert md[KEY]["indices"].tolist() == [0, 3, 4]
        assert md[KEY]["targets"].tolist() == [8, 10, 11]
        prepared = BaseTrainer.preforward(SimpleNamespace(device=torch.device("cpu"), LOG_SAMPLE=False), batch)
        torch.testing.assert_close(prepared["multimodal_metadata"][KEY]["targets"], md[KEY]["targets"])


@pytest.mark.skipif(not IS_MUSA_AVAILABLE, reason="Real TE checkpoint kernels require MUSA")
@pytest.mark.parametrize("backward_order", [(0, 1), (1, 0)])
def test_te_checkpoint_reuses_only_ephemeral_row_maps(monkeypatch, backward_order):
    from torch.utils.checkpoint import checkpoint

    from veomni.distributed.moe import deepep_ace as ace
    from veomni.distributed.moe import deepep_ace_checkpoint as cache

    class Event:
        def __init__(self):
            self.event = torch.musa.Event()
            self.event.record()

        def current_stream_wait(self):
            torch.musa.current_stream().wait_event(self.event)

    cpu_indices = (torch.arange(17)[:, None] + torch.arange(8)).remainder(32)
    counts = torch.bincount(cpu_indices.flatten(), minlength=32).tolist()
    indices = cpu_indices.to("musa")
    workspace = torch.empty(17, 2048, dtype=torch.bfloat16, device="musa")
    index_workspace = torch.empty_like(indices)

    class State:
        calls = 0
        num_experts = 64
        group = SimpleNamespace(size=lambda: 2)
        handle = ()
        recv_counts = counts
        combine_direct_write = False

        def dispatch(self, hidden, selected, probabilities, previous_event=None):
            State.calls += 1
            self.token_num = hidden.shape[0]
            workspace.copy_(hidden)
            index_workspace.copy_(selected)
            return workspace, index_workspace, probabilities.clone()

        def wait_dispatch(self):
            pass

        def reverse_combine(self, hidden, probabilities):
            return hidden, probabilities

    monkeypatch.setattr(ace, "_current_stream_event", Event)
    monkeypatch.setattr(
        cache,
        "get_ops_config",
        lambda: SimpleNamespace(moe_dispatcher="deepep_ace", moe_deepep_recompute_cache="compact"),
    )
    monkeypatch.setenv("VEOMNI_MUSA_ACE_TE_SLOT_PROBABILITY", "1")
    torch.manual_seed(49)
    inputs = [
        (
            torch.randn(17, 2048, device="musa", dtype=torch.bfloat16),
            torch.randn(17, 8, device="musa", dtype=torch.float32),
        )
        for _ in range(2)
    ]

    def block(hidden, probabilities):
        compact, probs, *_ = cache._DispatchCompact.apply(hidden.sin(), indices, probabilities, State())
        for record in cache._CHECKPOINT_DISPATCH_CACHE.get():
            assert "_forward_row_nt" not in record.plan and "_reuse_forward_row_nt" not in record.plan
            assert "row_nt" not in record.plan and "row_map" not in record.plan
        return compact * probs.to(compact.dtype).unsqueeze(-1)

    def run(setting):
        monkeypatch.setenv("VEOMNI_MUSA_ACE_COMPACT_STATE_REUSE", setting)
        State.calls = 0
        xs = [tuple(t.clone().requires_grad_() for t in sample) for sample in inputs]
        ys = [checkpoint(block, *x, use_reentrant=False, context_fn=cache.deepep_recompute_context_fn) for x in xs]
        workspace.fill_(999)
        index_workspace.fill_(-1)
        for i in backward_order:
            ys[i].float().square().sum().backward()
        assert State.calls == 2
        assert cache._CHECKPOINT_DISPATCH_CACHE.get() is None
        return ys + [t.grad for sample in xs for t in sample]

    expected, actual = run("0"), run("1")
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.skipif(not IS_MUSA_AVAILABLE, reason="Router payload conversion requires MUSA")
def test_router_payload_preserves_bf16_round_trip_and_gradient(monkeypatch):
    from functools import partial

    from torch.distributed.fsdp import MixedPrecisionPolicy
    from torch.distributed.fsdp._fully_shard._fsdp_common import _cast_fp_tensor
    from torch.distributed.utils import _apply_to_tensors

    from veomni.distributed import moe as moe_dispatch
    from veomni.distributed import parallel_state
    from veomni.models.transformers.qwen3_5_moe import musa_router_payload as payload
    from veomni.models.transformers.qwen3_5_moe.generated import patched_modeling_qwen3_5_moe_gpu as modeling
    from veomni.ops.config import singleton
    from veomni.ops.kernels.moe import _make_moe_experts_adapter, musa_group_gemm
    from veomni.ops.kernels.moe.routing_payload import expert_fsdp_kwargs

    torch.manual_seed(83)
    value = torch.randn(4096, device="musa", dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        value[:6] = torch.tensor([0.0, -0.0, 1e-40, -1e-40, 1.00390625, -1.00390625], device="musa")
    dy = torch.randn_like(value)
    expected = value.to(torch.bfloat16).float()
    actual = payload._RoundedPayload.apply(value)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual, value, dy)[0], torch.autograd.grad(expected, value, dy)[0], rtol=0, atol=0
    )
    monkeypatch.setenv("VEOMNI_MUSA_ROUTER_FP32_PAYLOAD", "1")
    config = SimpleNamespace(moe_dispatcher="deepep_ace", moe_implementation="fused_musa")
    state = SimpleNamespace(ep_enabled=True)
    monkeypatch.setattr(payload, "get_ops_config", lambda: config)
    monkeypatch.setattr(singleton, "get_ops_config", lambda: config)
    monkeypatch.setattr(payload, "get_parallel_state", lambda: state)
    monkeypatch.setattr(parallel_state, "get_parallel_state", lambda: state)
    monkeypatch.setattr(musa_group_gemm, "get_parallel_state", lambda: state)
    router = modeling.Qwen3_5MoeTopKRouter(SimpleNamespace(hidden_size=128, num_experts=32, num_experts_per_tok=8)).to(
        device="musa", dtype=torch.bfloat16
    )
    with torch.no_grad():
        router.weight.normal_(std=0.02)
    hidden = torch.randn(129, 128, device="musa", dtype=torch.bfloat16, requires_grad=True)
    original = type(router).forward
    candidate = payload.make_candidate(original)
    expected = original(router, hidden)
    actual = candidate(router, hidden)
    assert type(actual[1]) is torch.Tensor
    torch.testing.assert_close(actual[1], expected[1].float(), rtol=0, atol=0)
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    observed = []

    def dispatch(ep_class, num_experts, weights, indices, features, *parameters):
        observed.append((weights.dtype, weights.data_ptr()))
        return features[:, :8].float() * weights.float()

    monkeypatch.setattr(moe_dispatch, "dispatch_to_ep_class", dispatch)
    experts = SimpleNamespace(
        num_experts=32, gate_up_proj=torch.empty(1), down_proj=torch.empty(1), _veomni_qwen_rounded_payload=True
    )
    original_kwargs = dict(mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32))
    new_kwargs = expert_fsdp_kwargs(experts, original_kwargs)
    assert original_kwargs["mp_policy"].cast_forward_inputs
    assert not new_kwargs["mp_policy"].cast_forward_inputs
    assert new_kwargs["mp_policy"].reduce_dtype == torch.float32
    assert new_kwargs["mp_policy"].param_dtype == torch.bfloat16
    adapter = _make_moe_experts_adapter(musa_group_gemm.musa_fused_moe_forward, allow_ace_fp32_payload=True)
    original_inputs = _apply_to_tensors(partial(_cast_fp_tensor, torch.bfloat16), (hidden, expected[2], expected[1]))
    expected_out = adapter(experts, *original_inputs)
    cast_inputs = (hidden, actual[2], actual[1])
    actual_out = adapter(experts, *cast_inputs)
    assert observed[0][0] == torch.bfloat16
    assert observed[1] == (torch.float32, actual[1].data_ptr())
    torch.testing.assert_close(actual_out, expected_out, rtol=0, atol=0)
    dy = torch.randn_like(actual_out)
    for got, want in zip(
        torch.autograd.grad(actual_out, (hidden, router.weight), dy),
        torch.autograd.grad(expected_out, (hidden, router.weight), dy),
    ):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    fallback = _make_moe_experts_adapter(lambda **kwargs: kwargs["routing_weights"], allow_ace_fp32_payload=True)
    state.ep_enabled = False
    assert fallback(experts, *cast_inputs).dtype == torch.bfloat16
    state.ep_enabled = True
    config.moe_dispatcher = "alltoall"
    assert fallback(experts, *cast_inputs).dtype == torch.bfloat16
    config.moe_dispatcher = "deepep_ace"
    assert expert_fsdp_kwargs(SimpleNamespace(), original_kwargs) is original_kwargs
    hidden_fallback = _make_moe_experts_adapter(lambda **kwargs: kwargs, allow_ace_fp32_payload=True)
    fallback_inputs = hidden_fallback(experts, hidden.float(), actual[2], actual[1])
    assert fallback_inputs["hidden_states"].dtype == torch.bfloat16
    assert fallback_inputs["routing_weights"].dtype == torch.bfloat16
    monkeypatch.setattr(payload, "get_active_replay", lambda: object())
    assert candidate(router, hidden)[1].dtype == torch.bfloat16


@pytest.mark.parametrize("policy_kind", ["no_cast", "none", "fp32", "bf16"])
def test_router_payload_preserves_unrelated_fsdp_policies(monkeypatch, policy_kind):
    from torch.distributed.fsdp import MixedPrecisionPolicy

    from veomni.ops.kernels.moe import routing_payload

    monkeypatch.setattr(routing_payload, "qualified_experts", lambda module: True)
    module = SimpleNamespace()
    policy = (
        None
        if policy_kind == "none"
        else MixedPrecisionPolicy(
            param_dtype=torch.float32 if policy_kind == "fp32" else torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
            cast_forward_inputs=policy_kind != "no_cast",
        )
    )
    kwargs = {"mp_policy": policy}
    actual = routing_payload.expert_fsdp_kwargs(module, kwargs)
    if policy_kind == "bf16":
        assert actual is not kwargs and not actual["mp_policy"].cast_forward_inputs
        assert actual["mp_policy"].output_dtype == actual["mp_policy"].reduce_dtype == torch.float32
        assert kwargs["mp_policy"].cast_forward_inputs
        assert module._veomni_payload_input_dtype == torch.bfloat16
    else:
        assert actual is kwargs
        assert not hasattr(module, "_veomni_payload_input_dtype")


def test_plain_routing_payload_participates_in_fsdp_post_backward_gate():
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup

    hidden = torch.randn(3, 8, requires_grad=True)
    routing = torch.randn(3, 8, requires_grad=True)
    callbacks = []
    group = SimpleNamespace(post_backward=lambda: callbacks.append("post_backward"))
    (hooked_hidden, hooked_routing), kwargs = FSDPParamGroup._register_post_backward_hook(group, (hidden, routing), {})
    assert not kwargs
    assert hooked_hidden.grad_fn is hooked_routing.grad_fn
    assert "RegisterPostBackward" in type(hooked_hidden.grad_fn).__name__
    (hooked_hidden * hooked_routing).sum().backward()
    assert callbacks == ["post_backward"]
    torch.testing.assert_close(hidden.grad, routing)
    torch.testing.assert_close(routing.grad, hidden)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_packed_padding_preserves_logical_rows_and_zero_tail(dtype):
    from veomni.distributed.fsdp2 import musa_packed_padding as packed

    world, logical, padded = 8, 21, 24
    output = torch.empty(world * padded, dtype=dtype)
    expected = torch.arange(world * logical).reshape(world, logical).to(dtype)
    for rank in range(world):
        source, result = packed._packed_all_gather_copy_in(
            [expected[rank, :9], expected[rank, 9:]], output, [9, 12], padded, rank
        )
        assert source.numel() == padded and result is output
    rows = packed._logical_rows(output, world)
    torch.testing.assert_close(rows, expected, rtol=0, atol=0)
    assert not rows.is_contiguous()
    assert torch.count_nonzero(output.view(world, padded)[:, logical:]) == 0
    destinations = [torch.empty(world, 9, dtype=dtype), torch.empty(world, 12, dtype=dtype)]
    torch.ops.fsdp.split_with_sizes_copy(rows, [9, 12], dim=1, out=destinations)
    torch.testing.assert_close(torch.cat(destinations, dim=1), expected, rtol=0, atol=0)
    grads = [expected[:, :9].contiguous(), expected[:, 9:].contiguous()]
    reduce_input = torch.empty(world * padded, dtype=dtype)
    packed._packed_reduce_copy_in(grads, reduce_input, world)
    torch.testing.assert_close(reduce_input.view(world, padded)[:, :logical], expected, rtol=0, atol=0)
    assert torch.count_nonzero(reduce_input.view(world, padded)[:, logical:]) == 0
