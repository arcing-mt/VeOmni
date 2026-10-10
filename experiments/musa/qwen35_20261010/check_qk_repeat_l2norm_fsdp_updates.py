"""Exact first-order FSDP8/EP8 q/k preentry gate with native clip/Adam.

Four controlled projection/norm/value blocks. All outputs, gradients, clipped
gradients, norms, masters/states and update deltas must be bitwise equal.
No ACE/FA/real-Qwen or independent free-trajectory certification.
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

import qk_repeat_l2norm as prototype
import install_mccl_padding as padding


root = Path(__file__).parent
rank = int(os.environ["RANK"])
destination = root / f"qk_repeat_l2norm_fsdp_updates_rank{rank}.json"
report = {"status": "running", "rank": rank,
          "criteria": {"FW_exact": True, "native_repeat_exact": True,
                       "candidate_all_bitwise": True, "update_delta_exact": True},
          "results": [], "phase": "initialization",
          "source_sha256": {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
                            ("qk_repeat_l2norm.py", "qk_repeat_l2norm_kernel.py",
                             "check_qk_repeat_l2norm_fsdp_updates.py", "install_mccl_padding.py")}}


def persist():
    destination.write_text(json.dumps(report, indent=2) + "\n")


def fail_fast(kind, value, tb):
    report.update(status="failed", error="".join(traceback.format_exception(kind, value, tb)),
                  prototype_stats=dict(prototype.stats))
    persist()
    traceback.print_exception(kind, value, tb)
    sys.stderr.flush()
    os._exit(1)


sys.excepthook = fail_fast
persist()
torch.musa.set_device(int(os.environ["LOCAL_RANK"]))
prototype.initialize()
dist.init_process_group("mccl", device_id=torch.device("musa", int(os.environ["LOCAL_RANK"])))
ps = _init_parallel_state(dp_size=8, dp_shard_size=8, extra_parallel_sizes=(8,), name="base")
apply_musa_fsdp2_clip_grad_norm_patch()
torch.manual_seed(29043)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(512, 8192, bias=False)
        self.out = torch.nn.Linear(12288, 512, bias=False)
        self.extra = torch.nn.Parameter(torch.randn(513) * .01)
        self.expert = torch.nn.Linear(512, 512, bias=False)
        self.candidate = False

    def forward(self, x, cu, cu_cpu):
        packed = self.proj(x)
        q, k, v = packed.split((2048, 2048, 4096), dim=-1)
        q, k = (t.reshape(1, packed.shape[1], 16, 128) for t in (q, k))
        fn = prototype.repeat_norm if self.candidate else prototype.native
        q, k = fn(q), fn(k)
        # Exercise a nonzero value gradient as well as both normalized q/k heads.
        result = torch.cat((q.flatten(2), k.flatten(2), v), dim=-1)
        # A channel bias plus shared offset participates in FW, gradients and Adam.
        # Its non-aligned shard exercises the same padding wrapper as training.
        dense = self.out(result) + self.extra[:512] + self.extra[-1]
        return x + dense * .05 + self.expert(x) * .05


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block() for _ in range(4)])
        self.checkpoint = False

    def forward(self, x, cu, cu_cpu):
        for block in self.blocks:
            if self.checkpoint:
                x = checkpoint(block, x, cu, cu_cpu, use_reentrant=False)
            else:
                x = block(x, cu, cu_cpu)
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
    delta = a.double() - b.double()
    return {"equal": torch.equal(a, b), "rms":
            (delta.square().mean().sqrt() / b.double().square().mean().sqrt().clamp_min(1e-12)).item(),
            "max_abs": delta.abs().max().item()}


def run_fb(runtime, base_x, cu, cu_cpu):
    x = base_x.detach().requires_grad_()
    runtime.optimizer.zero_grad()
    before = [local(p)._version for p in runtime.model.parameters()]
    loss, output = runtime.model(x, cu, cu_cpu)
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
        layout = [0, 1, 4, 67, 131, 196] if step != 2 else [0, 313, 1090, 2048]
        x = torch.randn(1, layout[-1], 512, device="musa")
        cu_cpu = torch.tensor(layout, dtype=torch.int32)
        cu = cu_cpu.musa()
        loss0, out0, dx0, versions0 = run_fb(runtimes[0], x, cu, cu_cpu)
        raw0 = [snapshot(p.grad) for p in a.parameters()]
        loss1, out1, dx1, versions1 = run_fb(runtimes[0], x, cu, cu_cpu)
        repeats = {"loss": compare(loss1, loss0), "output": compare(out1, out0), "input_gradient": compare(dx1, dx0),
                   "raw_gradients": {name: compare(p.grad, old) for name, p, old in zip(names, a.parameters(), raw0)}}
        row = {"checkpoint": checkpoint_enabled, "step": step, "max_norm": maximum,
               "layout": layout, "native_repeat": repeats, "native_version_deltas": [versions0, versions1]}
        report["results"].append(row)
        persist()
        assert repeats["loss"]["equal"] and repeats["output"]["equal"] and repeats["input_gradient"]["equal"]
        assert all(v["equal"] for v in repeats["raw_gradients"].values()), repeats
        fused_before = prototype.stats["fused_backward"]
        loss2, out2, dx2, versions2 = run_fb(runtimes[1], x, cu, cu_cpu)
        row.update(candidate_fused_delta=prototype.stats["fused_backward"] - fused_before,
                   loss=compare(loss2, loss1), output=compare(out2, out1), input_gradient=compare(dx2, dx1),
                   raw_gradients={name: compare(pb.grad, pa.grad) for name, pa, pb in zip(names, a.parameters(), b.parameters())},
                   candidate_version_deltas=versions2)
        persist()
        assert row["candidate_fused_delta"] == 8 and prototype.stats["fallback"] == 0
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
        print("paired_case", rank, checkpoint_enabled, step, "norms", norms, "fused", row["candidate_fused_delta"], flush=True)

assert padding.calls["all_gather"] > 0 and padding.calls["reduce_scatter"] > 0
report.update(status="passed", phase="complete", prototype_stats=dict(prototype.stats),
              padding_stats=padding.stats(),
              scope="Four controlled projection/q-k-repeat-norm/value blocks; FSDP8/EP8 parameter groups, native depth-one prefetch, "
                    "padding/nativeAVG, BF16 compute and FP32 storage/reduction, native clip and dual fused AdamW. "
                    "Same-master paired inputs at every step; moments retained. CKPT False/True and three clip levels. "
                    "All outputs/gradients/norms/clipped gradients/master/step/moments and update deltas must be bitwise equal. "
                    "No ACE routing, FA, actual full Qwen weights or full-model trajectory certification.")
persist()
dist.destroy_process_group()
