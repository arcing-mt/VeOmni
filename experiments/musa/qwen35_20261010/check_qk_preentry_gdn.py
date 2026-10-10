"""Actual Qwen GDN class gate; random weights, first-order, no whole-model claim."""
from collections import Counter
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import traceback

import torch
import torch_musa
from torch.utils.checkpoint import checkpoint
import qk_repeat_l2norm as prototype

import install_qk_preentry as adapter

root = Path(__file__).parent
names = ("install_qk_preentry.py", "qk_repeat_l2norm.py", "qk_repeat_l2norm_kernel.py", "check_qk_preentry_gdn.py")
report = {"status": "running", "phase": "initialization", "criteria": {"native_repeat_all_bitwise": True, "candidate_all_bitwise": True},
          "results": [], "source_sha256": {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names},
          "scope": "Actual Qwen3_5MoeGatedDeltaNet with actual local model text config and random BF16 weights, "
                   "native FLA convolution/fused gated RMSNorm/MUSA TileLang GDN, CP off/on and original instance backend binding. "
                   "Deterministic algorithms enabled for this numerical gate only, because default native Linear dW is not bitwise repeatable. "
                   "No default-mode all-gradient, FA/ACE/FSDP/full-model/free-history or performance certification."}


def persist():
    (root / "qk_preentry_gdn_gate.json").write_text(json.dumps(report, indent=2) + "\n")


def compare(actual, expected):
    a, b = actual.detach().cpu(), expected.detach().cpu()
    assert a.shape == b.shape and a.dtype == b.dtype
    d = a.double() - b.double()
    return {"equal": torch.equal(a, b), "max_abs": d.abs().max().item(),
            "rms": (d.square().mean().sqrt()/b.double().square().mean().sqrt().clamp_min(1e-12)).item()}


def run_fb(model, x, dy, cu, checkpoint_enabled):
    model.zero_grad(set_to_none=True)
    inp = x.detach().requires_grad_()
    before = {name: p._version for name, p in model.named_parameters()}
    fn = lambda h: model(h, cu_seq_lens_q=cu)
    out = checkpoint(fn, inp, use_reentrant=False) if checkpoint_enabled else fn(inp)
    out.backward(dy)
    torch.musa.synchronize()
    assert all(p.grad is not None for p in model.parameters())
    return {"output": out.detach().cpu(), "input_gradient": inp.grad.detach().cpu(),
            "parameters": {name: p.grad.detach().cpu() for name, p in model.named_parameters()},
            "versions": {name: p._version-before[name] for name, p in model.named_parameters()}}


def compare_fb(actual, expected):
    return {"output": compare(actual['output'], expected['output']),
            "input_gradient": compare(actual['input_gradient'], expected['input_gradient']),
            "parameters": {name: compare(value, expected['parameters'][name]) for name, value in actual['parameters'].items()},
            "versions_equal": actual['versions'] == expected['versions']}


def exact(errors):
    return errors['output']['equal'] and errors['input_gradient']['equal'] and errors['versions_equal'] and all(x['equal'] for x in errors['parameters'].values())


try:
    persist()
    torch.musa.set_device(0)
    torch.use_deterministic_algorithms(True)
    report['deterministic_algorithms'] = torch.are_deterministic_algorithms_enabled()
    torch.manual_seed(49251)
    module = importlib.import_module(adapter.MODULE)
    for op, implementation in (("causal_conv1d", "fla"), ("rms_norm_gated", "fla"), ("chunk_gated_delta_rule", "musa_tilelang")):
        slot = getattr(module, f'veomni_{op}')
        slot.bind(implementation)
        assert slot.use_non_eager_impl and slot.bound_kernel() is not None
    config_path = Path("/data/share/models/Qwen3.5-35B-A3B/config.json")
    raw = json.loads(config_path.read_text())
    config = module.Qwen3_5MoeTextConfig(**raw['text_config'])
    report['model_config_sha256'] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    report['geometry'] = {name: getattr(config, name) for name in ('hidden_size', 'linear_num_key_heads', 'linear_num_value_heads', 'linear_key_head_dim', 'linear_value_head_dim', 'linear_conv_kernel_dim')}
    a = module.Qwen3_5MoeGatedDeltaNet(config, layer_idx=0).to(device='musa', dtype=torch.bfloat16)
    b = deepcopy(a)
    a._qk_preentry_native_only = True
    native_backend = a.chunk_gated_delta_rule
    assert native_backend is module.veomni_chunk_gated_delta_rule.bound_kernel()
    assert native_backend is importlib.import_module('veomni.ops.kernels.gated_delta_rule.musa_tilelang').chunk_gated_delta_rule
    assert a.causal_conv1d_fn is module.veomni_causal_conv1d.bound_kernel()
    assert type(a.norm) is module.veomni_rms_norm_gated.bound_kernel()
    adapter.install()
    adapter.install()
    assert a.chunk_gated_delta_rule is native_backend and b.chunk_gated_delta_rule is native_backend
    # Rebinding the global slot must not alter either existing instance.
    slot = module.veomni_chunk_gated_delta_rule
    slot.bind('eager')
    assert not slot.use_non_eager_impl and slot.bound_kernel() is None
    assert a.chunk_gated_delta_rule is native_backend and b.chunk_gated_delta_rule is native_backend
    report['cached_instance_independence'] = True
    # Fresh process: original native L2norm autotuner has no in-memory config.
    # Do not clear/mutate it. First CP forward falls back, recompute may become
    # fused; original and recomputed saved-tensor contracts must still agree.
    probe = torch.empty(1, 256, 8192, device='musa', dtype=torch.bfloat16)
    probe_q = probe[..., :2048].reshape(1, 256, 16, 128)
    assert prototype.native_config(probe_q) is None
    del probe_q, probe
    x = torch.randn(1, 256, config.hidden_size, dtype=torch.bfloat16, device='musa') * .1
    dy = torch.randn_like(x) * .01
    cu = torch.tensor([0, 128, 256], dtype=torch.int32, device='musa')
    report['phase'] = 'natural_cold_checkpoint_first_candidate'
    before = adapter.stats()
    cold = run_fb(b, x, dy, cu, True)
    after = adapter.stats()
    expected = run_fb(a, x, dy, cu, True)
    errors = compare_fb(cold, expected)
    report['natural_cold_checkpoint'] = {'candidate': errors, 'before': before, 'after': after}
    persist()
    assert exact(errors), errors
    # q warms the shared D/NB/dtype key; k is already warm in the first FW.
    assert after['prototype'].get('native_config_fallback', 0)-before['prototype'].get('native_config_fallback', 0) == 1
    assert after['prototype'].get('fused_forward', 0)-before['prototype'].get('fused_forward', 0) == 3
    assert after['prototype'].get('fused_backward', 0)-before['prototype'].get('fused_backward', 0) == 1
    del x, dy, cu, cold, expected
    for cp in (False, True):
        for layout in ([0, 128, 256], [0, 128, 385, 2048], [0, 313, 1090, 2861, 4890, 6475, 8155]):
            report['phase'] = f'cp={cp}, S={layout[-1]}'
            persist()
            x = torch.randn(1, layout[-1], config.hidden_size, dtype=torch.bfloat16, device='musa') * .1
            dy = torch.randn_like(x) * .01
            cu = torch.tensor(layout, dtype=torch.int32, device='musa')
            reference0 = run_fb(a, x, dy, cu, cp)
            reference = run_fb(a, x, dy, cu, cp)
            repeats = compare_fb(reference, reference0)
            row = {'checkpoint': cp, 'layout': layout, 'native_repeat': repeats}
            report['results'].append(row)
            persist()
            assert exact(repeats), repeats
            before = adapter.stats()
            actual = run_fb(b, x, dy, cu, cp)
            errors = compare_fb(actual, reference)
            after = adapter.stats()
            row.update(candidate=errors, stats_before=before, stats_after=after,
                       fused_backward_delta=after['prototype'].get('fused_backward',0)-before['prototype'].get('fused_backward',0))
            persist()
            assert exact(errors), errors
            assert row['fused_backward_delta'] == 2
            assert after['adapter'].get('preentry_fast',0)-before['adapter'].get('preentry_fast',0) >= 1
            assert after['adapter'].get('preentry_native_layout',0) == before['adapter'].get('preentry_native_layout',0)
            print('component_passed', cp, layout[-1], row['fused_backward_delta'], flush=True)
            del x, dy, cu, reference0, reference, actual
    # A proxy instance backend is deliberately unsupported even though it delegates
    # to the same native callable: use the complete original decorated forward.
    x = torch.randn(1, 256, config.hidden_size, dtype=torch.bfloat16, device='musa') * .1
    dy = torch.randn_like(x) * .01
    cu = torch.tensor([0, 128, 256], dtype=torch.int32, device='musa')
    reference = run_fb(a, x, dy, cu, False)
    b.chunk_gated_delta_rule = lambda *args, **kwargs: native_backend(*args, **kwargs)
    before = adapter.stats()
    actual = run_fb(b, x, dy, cu, False)
    after = adapter.stats()
    errors = compare_fb(actual, reference)
    report['backend_proxy_fallback'] = {'errors': errors, 'before': before, 'after': after}
    assert exact(errors)
    assert after['adapter'].get('original_forward',0) == before['adapter'].get('original_forward',0)+1
    assert after['prototype'].get('fused_forward',0) == before['prototype'].get('fused_forward',0)
    b.chunk_gated_delta_rule = native_backend
    before = adapter.stats()
    with torch.no_grad():
        expected = a(x, cu_seq_lens_q=cu)
        actual = b(x, cu_seq_lens_q=cu)
    after = adapter.stats()
    report['no_grad_fallback'] = {'error': compare(actual, expected), 'before': before, 'after': after}
    assert report['no_grad_fallback']['error']['equal']
    assert after['adapter'].get('original_forward',0) == before['adapter'].get('original_forward',0)+2
    assert after['prototype'].get('fused_forward',0) == before['prototype'].get('fused_forward',0)
    assert adapter.counts['installed'] == 1
    report.update(status='passed', adapter_stats=adapter.stats())
except BaseException:
    report.update(status='failed', error=traceback.format_exc(), adapter_stats=adapter.stats())
    persist()
    raise
finally:
    persist()
