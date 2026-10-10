"""Controlled composed integer-count/slot-copy TE FSDP8/EP8 native clip/Adam gate.

Four elementwise hidden/router/expert blocks exercise both compactor adjoints
and unchanged native TE merging. No ACE/FA/full-Qwen trajectory certification.
"""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

import torch
import torch_musa
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.utils.checkpoint import checkpoint
from veomni.distributed.parallel_state import _init_parallel_state
from veomni.models.model_runtime import VeOmniModelRuntime
from veomni.ops.platform.musa import apply_musa_fsdp2_clip_grad_norm_patch
from veomni.optim.optimizer import build_optimizer, MultiOptimizer

import install_te_compact
import install_te_counts_slot
import te_slot_probability as slot_prototype
import te_compact_prototype as reference
import install_te_counts_slot as prototype
import install_mccl_padding as padding


root = Path(__file__).parent
rank = int(os.environ["RANK"])
destination = root / f"te_counts_slot_fsdp_updates_rank{rank}.json"
report = {"status": "running", "rank": rank,
          "criteria": {"FW_exact": True, "native_repeat_exact": True,
                       "candidate_all_bitwise": True, "update_delta_exact": True},
          "results": [], "phase": "initialization",
          "source_sha256": {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
                            ("install_te_counts_slot.py", "install_te_native_counts.py", "te_slot_probability.py",
                             "check_te_counts_slot_fsdp_updates.py", "te_compact_prototype.py",
                             "install_te_compact.py", "install_mccl_padding.py")}}


def persist():
    destination.write_text(json.dumps(report, indent=2) + "\n")


def fail_fast(kind, value, tb):
    report.update(status="failed", error="".join(traceback.format_exception(kind, value, tb)),
                  prototype_stats=prototype.stats())
    persist()
    traceback.print_exception(kind, value, tb)
    sys.stderr.flush()
    os._exit(1)


sys.excepthook = fail_fast
persist()
gate_path = Path(os.environ["TE_COUNTS_SLOT_GATE_REPORT"])
gate = json.loads(gate_path.read_text())
assert gate["status"] == "passed_operator_gates"
for name, digest in gate["source_sha256"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
report["operator_gate_sha256"] = hashlib.sha256(gate_path.read_bytes()).hexdigest()
report["operator_gate_source_sha256"] = gate["source_sha256"]
micro_path = Path(os.environ["TE_COUNTS_SLOT_ACTUAL_MICRO"])
micro = json.loads(micro_path.read_text())
assert micro["status"] == "completed_actual_routing_micro" and len(micro["cases"]) == 10
assert micro["gate_sha256"] == report["operator_gate_sha256"]
assert all(row["measurements"][1]["median_ms"] < min(row["measurements"][0]["median_ms"], row["measurements"][2]["median_ms"]) for row in micro["cases"])
for name, digest in micro["source_sha256"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
report["actual_micro_sha256"] = hashlib.sha256(micro_path.read_bytes()).hexdigest()
report["deterministic_algorithms_enabled"] = torch.are_deterministic_algorithms_enabled()
assert not report["deterministic_algorithms_enabled"]
torch.musa.set_device(int(os.environ["LOCAL_RANK"]))
os.environ["TE_COMPACT"] = "1"
os.environ["TE_NATIVE_LONG_COUNTS"] = "1"
os.environ["TE_SLOT_PROBABILITY"] = "1"
os.environ["TE_COUNTS_SLOT_COMBINED"] = "1"
install_te_compact.load_native()
# Intended original TE control: authoritative CPU integer counts, same native floating math.
import ast
import inspect
control_tree = ast.parse(inspect.getsource(reference._te_compact_permute))
control_fn = control_tree.body[0]
control_fn.name = "_intended_native_TE_counts"
target = ast.dump(ast.parse("counts = multi_hot.sum(0, dtype=torch.long)").body[0])
found = [i for i, node in enumerate(control_fn.body) if ast.dump(node) == target]
assert len(found) == 1
control_fn.body[found[0]] = ast.parse("counts = torch.as_tensor(expert_counts, device=recv_hidden.device, dtype=torch.long)").body[0]
ast.fix_missing_locations(control_tree)
control_namespace = dict(reference.__dict__)
exec(compile(control_tree, "<intended_native_TE_CPU_counts>", "exec"), control_namespace)
original_compact = control_namespace[control_fn.name]
assert install_te_counts_slot.install()
candidate_compact = reference._te_compact_permute
dist.init_process_group("mccl", device_id=torch.device("musa", int(os.environ["LOCAL_RANK"])))
assert dist.get_world_size() == 8
ps = _init_parallel_state(dp_size=8, dp_shard_size=8, extra_parallel_sizes=(8,), name="base")
apply_musa_fsdp2_clip_grad_norm_patch()
torch.manual_seed(29043)


class ExpertScale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2048))
        self.bias = torch.nn.Parameter(torch.randn(2049) * .01)

    def forward(self, x):
        return x * self.weight + self.bias[:2048] + self.bias[-1]


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(2048))
        self.bias = torch.nn.Parameter(torch.randn(2049) * .01)
        self.route = torch.nn.Parameter(torch.randn(8) * .01)
        # A learned table participates in FW/BW and every Adam comparison.
        # Its >1M local shards plus bias/router tails exercise padding.
        self.table = torch.nn.Parameter(torch.randn(32705, 2048) * .01)
        self.expert = ExpertScale()
        self.candidate = False

    def forward(self, x, indices, counts):
        assert x.shape[0] <= 32704
        hidden = (x * self.scale + self.bias[:2048] + self.bias[-1]
                  + self.table[:x.shape[0]] * .01 + self.table[-1] * .01).contiguous()
        probabilities = torch.softmax(x[:, :8].float() * .01 + self.route.float(), -1)
        fn = candidate_compact if self.candidate else original_compact
        result = fn(hidden, indices, probabilities, 32, counts)
        assert result is not None, reference.last_fallback_reason
        y, p, _, _, row_map, width = result
        expert = self.expert(y)
        weighted = expert * p.to(expert.dtype).unsqueeze(-1)
        merged = reference._TECompactUnpermute.apply(weighted, row_map, width, x.shape[0])
        return x + merged * .05


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block() for _ in range(4)])
        self.checkpoint = False

    def forward(self, x, indices, counts):
        for block in self.blocks:
            if self.checkpoint:
                x = checkpoint(block, x, indices, counts, use_reentrant=False,
                               determinism_check="default", preserve_rng_state=True)
            else:
                x = block(x, indices, counts)
        return x.float().square().mean(), x


a = Model().to("musa")
b = deepcopy(a)
for block in b.blocks:
    block.candidate = True
policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
runtimes = []
for model in (a, b):
    ep = []
    for block in model.blocks:
        fully_shard(block.expert, mesh=ps.extra_parallel_fsdp_device_mesh["ep"]["ep_fsdp"],
                    reshard_after_forward=True, mp_policy=policy)
        fully_shard(block, mesh=ps.fsdp_mesh, reshard_after_forward=True, mp_policy=policy)
        block._fsdp_modules = [block.expert, block]
        ep.extend(block.expert.parameters())
    fully_shard(model, mesh=ps.fsdp_mesh, reshard_after_forward=True, mp_policy=policy)
    for index, block in enumerate(model.blocks):
        block.set_modules_to_forward_prefetch(list(reversed(model.blocks[index+1]._fsdp_modules)) if index < 3 else [])
        block.set_modules_to_backward_prefetch(list(reversed(model.blocks[index-1]._fsdp_modules)) if index > 0 else [])
    ids = {id(p) for p in ep}
    dense = [p for p in model.parameters() if id(p) not in ids]
    model._extra_parallel_param_groups = {"ep": ep, "non_extra_parallel": dense}
    runtime = VeOmniModelRuntime.__new__(VeOmniModelRuntime)
    runtime.model, runtime.model_name = model, "base"
    runtime.args = SimpleNamespace(optimizer=SimpleNamespace(max_grad_norm=1.))
    runtime.optimizer = build_optimizer(model, lr=1e-4, weight_decay=.1, fused=True)
    assert isinstance(runtime.optimizer, MultiOptimizer)
    assert set(runtime.optimizer.optimizers_dict) == {"ep", "non_extra_parallel"}
    assert all(isinstance(opt, torch.optim.AdamW) for opt in runtime.optimizer.optimizers_dict.values())
    assert all(g["fused"] for opt in runtime.optimizer.optimizers_dict.values() for g in opt.param_groups)
    runtimes.append(runtime)
padding.install()


def local(t):
    return t.to_local() if hasattr(t, "to_local") else t


def snapshot(t):
    return local(t).detach().cpu().clone()


def compare(actual, expected):
    a, b = snapshot(actual), snapshot(expected)
    assert a.shape == b.shape and a.dtype == b.dtype
    if a.is_floating_point():
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        integer_dtype = {torch.float32: torch.int32, torch.float64: torch.int64,
                         torch.bfloat16: torch.int16, torch.float16: torch.int16}[a.dtype]
        exact_bits = torch.equal(a.contiguous().view(integer_dtype), b.contiguous().view(integer_dtype))
    else:
        exact_bits = torch.equal(a, b)
    delta = a.double() - b.double()
    return {"equal": exact_bits, "numeric_equal": torch.equal(a, b), "rms":
            (delta.square().mean().sqrt() / b.double().square().mean().sqrt().clamp_min(1e-12)).item(),
            "max_abs": delta.abs().max().item()}


def run_fb(runtime, base_x, indices, counts):
    x = base_x.detach().requires_grad_()
    runtime.optimizer.zero_grad()
    before = [local(p)._version for p in runtime.model.parameters()]
    loss, output = runtime.model(x, indices, counts)
    loss.backward()
    after = [local(p)._version for p in runtime.model.parameters()]
    return snapshot(loss), snapshot(output), snapshot(x.grad), [y-x for x, y in zip(before, after)]


def optimizer_state(runtime, p):
    return next(opt.state[p] for opt in runtime.optimizer.optimizers_dict.values() if p in opt.state)


names = [name for name, _ in a.named_parameters()]
assert names == [name for name, _ in b.named_parameters()]
for checkpoint_enabled in (False, True):
    a.checkpoint = b.checkpoint = checkpoint_enabled
    for step, maximum in enumerate((1., .001, 100.)):
        report["phase"] = f"checkpoint={checkpoint_enabled}, step={step}, norm={maximum}"
        with torch.no_grad():
            for pa, pb in zip(a.parameters(), b.parameters()):
                local(pb).copy_(local(pa))
        torch.manual_seed(39031 + rank*10 + step)
        tokens = (24577, 27988, 32680)[step]
        x = torch.randn(tokens, 2048, device="musa")
        ids = (torch.arange(tokens, device="cpu")[:, None] + torch.arange(8, device="cpu")[None, :] * 3 + rank) % 32
        ids[(torch.arange(tokens, device="cpu")[:, None] + torch.arange(8, device="cpu")[None, :] * 2) % 5 == 0] = -1
        ids[::17] = -1
        counts = torch.bincount(ids[ids >= 0], minlength=32).tolist()
        indices = ids.musa()
        loss0, out0, dx0, versions0 = run_fb(runtimes[0], x, indices, counts)
        raw0 = [snapshot(p.grad) for p in a.parameters()]
        loss1, out1, dx1, versions1 = run_fb(runtimes[0], x, indices, counts)
        repeats = {"loss": compare(loss1, loss0), "output": compare(out1, out0), "input_gradient": compare(dx1, dx0),
                   "raw_gradients": {name: compare(p.grad, old) for name, p, old in zip(names, a.parameters(), raw0)}}
        row = {"checkpoint": checkpoint_enabled, "step": step, "max_norm": maximum,
               "tokens": tokens, "native_repeat": repeats, "native_version_deltas": [versions0, versions1]}
        report["results"].append(row)
        persist()
        assert repeats["loss"]["equal"] and repeats["output"]["equal"] and repeats["input_gradient"]["equal"]
        assert all(v["equal"] for v in repeats["raw_gradients"].values()), repeats
        assert versions0 == versions1, row
        fused_before = dict(prototype.COUNTERS)
        slot_before = dict(slot_prototype.COUNTERS)
        loss2, out2, dx2, versions2 = run_fb(runtimes[1], x, indices, counts)
        row.update(candidate_counter_delta={key: value - fused_before.get(key, 0) for key, value in prototype.COUNTERS.items() if value != fused_before.get(key, 0)},
                   loss=compare(loss2, loss1), output=compare(out2, out1), input_gradient=compare(dx2, dx1),
                   raw_gradients={name: compare(pb.grad, pa.grad) for name, pa, pb in zip(names, a.parameters(), b.parameters())},
                   candidate_version_deltas=versions2)
        persist()
        assert row["candidate_counter_delta"] == {"Long_input_column_sum": 8 if checkpoint_enabled else 4}, row
        row["slot_candidate_delta"] = {key: value - slot_before.get(key, 0) for key, value in slot_prototype.COUNTERS.items() if value != slot_before.get(key, 0)}
        assert row["slot_candidate_delta"] == {"fused_weights": 8 if checkpoint_enabled else 4, "direct_slot_gradient": 4}, row
        assert row["loss"]["equal"] and row["output"]["equal"] and row["input_gradient"]["equal"]
        assert all(v["equal"] for v in row["raw_gradients"].values()), row
        assert versions1 == versions2, row
        norms = [float(runtime.clip_grad_norm(maximum)) for runtime in runtimes]
        row.update(norms=norms, norm_rms=abs(norms[1]-norms[0])/max(abs(norms[0]), 1e-12),
                   clipped_gradients={name: compare(pb.grad, pa.grad) for name, pa, pb in zip(names, a.parameters(), b.parameters())})
        persist()
        assert norms[0] == norms[1] and all(v["equal"] for v in row["clipped_gradients"].values())
        before_step = [[snapshot(p) for p in runtime.model.parameters()] for runtime in runtimes]
        for runtime in runtimes:
            runtime.optimizer.step()
        states = {}
        update_diagnostics = {}
        for index, (name, pa, pb) in enumerate(zip(names, a.parameters(), b.parameters())):
            sa, sb = optimizer_state(runtimes[0], pa), optimizer_state(runtimes[1], pb)
            states[name] = {"master": compare(pb, pa), **{key: compare(sb[key], sa[key]) for key in
                                                         ("step", "exp_avg", "exp_avg_sq")}}
            update_diagnostics[name] = compare(snapshot(pb) - before_step[1][index],
                                               snapshot(pa) - before_step[0][index])
        row["optimizer_states"] = states
        row["update_deltas"] = update_diagnostics
        persist()
        assert all(v["equal"] for state in states.values() for v in state.values()), states
        assert all(v["equal"] for v in update_diagnostics.values()), update_diagnostics
        assert all(local(p).dtype == torch.float32 and local(p.grad).dtype == torch.float32 for p in b.parameters())
        print("paired_case", rank, checkpoint_enabled, step, "norms", norms, "fused", row["candidate_counter_delta"], flush=True)

assert padding.calls["all_gather"] > 0 and padding.calls["reduce_scatter"] > 0
assert all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
           for name, digest in report["source_sha256"].items())
report.update(status="passed", phase="complete", prototype_stats=prototype.stats(),
              padding_stats=padding.stats(),
              scope="Four controlled hidden/router/elementwise-expert blocks with composed counts/slot-copy TE compaction and unchanged native weighted merge; "
                    "FSDP8/EP8 parameter groups, native depth-one prefetch, padding/nativeAVG, BF16 compute/FP32 storage/reduction, "
                    "native clip and dual fused AdamW. Same-master paired inputs at every step; moments retained. "
                    "CKPT False/True, N24577/27988/32680, invalid/unique routing and three clip levels. "
                    "All outputs/gradients/norms/clipped gradients/master/step/moments/update deltas must be bitwise equal. "
                    "Intended original TE math with authoritative CPU integer counts is the numerical control; no bitwise equivalence to old BF16 atomic fallback. No ACE collective, real expert MLP, FA, actual full Qwen or independent free-trajectory certification.")
persist()
dist.destroy_process_group()
