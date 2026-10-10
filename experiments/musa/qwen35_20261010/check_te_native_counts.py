"""Integer workaround gate against intended TE math and real native fallback."""
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import numpy as np
import torch
import torch_musa
from veomni.distributed.moe import deepep_ace
import install_te_compact
import install_te_native_counts as trial
import te_compact_prototype as reference

REPORT = {"status": "running", "cases": [], "comparisons": [], "benchmarks": []}
ROOT = Path(__file__).resolve().parent
DESTINATION = Path(os.environ["TE_NATIVE_COUNTS_GATE_REPORT"])


def persist(stage):
    REPORT["stage"] = stage
    DESTINATION.write_text(json.dumps(REPORT, indent=2) + "\n")
    print(stage, flush=True)


def exact(a, b, label):
    a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    item = {"stage": REPORT["stage"], "label": label, "shape": list(a.shape),
            "shape_dtype_equal": a.shape == b.shape and a.dtype == b.dtype}
    REPORT["comparisons"].append(item)
    assert item["shape_dtype_equal"], item
    if a.dtype.is_floating_point:
        dt = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
        a, b = a.view(dt), b.view(dt)
    item["raw_equal"] = bool(torch.equal(a, b))
    item["mismatches"] = int(torch.count_nonzero(a != b))
    assert item["raw_equal"], item


def rms(a, gold):
    a = a.detach().cpu().double()
    return float(((a - gold).square().sum() / gold.square().sum().clamp_min(1e-30)).sqrt())


def metadata(n):
    t, k = torch.arange(n, device="cpu")[:, None], torch.arange(8, device="cpu")[None, :]
    ids = (t + k * 3) % 32
    ids[(t + k * 2) % 5 == 0] = -1
    ids[::17] = -1
    return ids


def counts_and_slots(ids):
    valid = torch.nonzero(ids.flatten() >= 0).flatten()
    slots = valid[torch.argsort(ids.flatten()[valid], stable=True)]
    counts = torch.bincount(ids[ids >= 0], minlength=32)
    return counts, slots


def main():
    names = (Path(__file__).name, "install_te_native_counts.py", "te_compact_prototype.py", "install_te_compact.py")
    frozen = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
    REPORT.update(source_sha256=frozen, torch=torch.__version__, torch_musa=torch_musa.__version__,
                  deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                  scope="Default-mode GPU0 operator/complete-helper gate. Intended TE control has "
                        "authoritative CPU integer counts; broken original None is not numeric control. "
                        "Actual native fallback is only an independent timing baseline. No FSDP/model certification.",
                  raw_FP64_relative_RMS_threshold=0.01)
    assert not REPORT["deterministic_algorithms"]
    REPORT["native_counting_sort_env"] = os.environ.get("VEOMNI_MUSA_DEEPEP_COUNTING_SORT")
    assert REPORT["native_counting_sort_env"] == "1", "Timing must use production native stable-slot fallback"
    from veomni.ops.kernels.moe import musa_deepep_compact
    REPORT["native_fallback_source_sha256"] = {
        str(Path(m.__file__)): hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
        for m in (deepep_ace, musa_deepep_compact)}
    torch.musa.set_device(0)
    torch.manual_seed(4407)
    install_te_compact.load_native()
    native_fallback = deepep_ace._compact_permute
    original = reference._te_compact_permute
    os.environ.pop("TE_NATIVE_LONG_COUNTS", None)
    assert not trial.install() and reference._te_compact_permute is original
    REPORT["default_off_checked"] = True
    os.environ["TE_COMPACT"] = "1"
    os.environ["TE_NATIVE_LONG_COUNTS"] = "1"
    assert trial.install()
    candidate = reference._te_compact_permute
    # Restore the single changed assignment in a syntax proof.
    old_tree = ast.parse(inspect.getsource(original))
    fn = old_tree.body[0]
    count_index = next(i for i, node in enumerate(fn.body)
                       if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                       and node.targets[0].id == "counts")
    # Intended TE numerical control: only substitute authoritative CPU counts.
    fn.name = "_intended_te_with_cpu_counts"
    fn.body[count_index] = ast.parse("counts = _gate_cpu_counts.to(device=recv_hidden.device)").body[0]
    ast.fix_missing_locations(old_tree)
    control_namespace = dict(reference.__dict__)
    exec(compile(old_tree, "<intended_original_TE_CPU_integer_counts>", "exec"), control_namespace)
    control = control_namespace[fn.name]
    REPORT["numeric_control"] = "Original TE floating kernels/class/weights/maps/weighted merge, authoritative CPU counts copied to GPU; not broken original fallback."
    proof = json.loads(Path(os.environ["TE_COUNTS_PROOF"]).read_text())
    actual_root = Path(os.environ["TE_COUNTS_ACTUAL_RAW"])
    actual_manifest = json.loads((ROOT / "te_native_counts_operator_actual_sha256.json").read_text())
    for name, expected in actual_manifest.items():
        assert hashlib.sha256((actual_root / name).read_bytes()).hexdigest() == expected, name
    REPORT["actual_file_sha256"] = actual_manifest
    actual_ids = []
    for item in proof["samples"]:
        # The original same-call proof identifies each file and its digest.
        rank, category = item["rank"], item["category"]
        files = list(actual_root.glob(f"te_native_counts_rank{rank}_{category}_indices.npy"))
        assert len(files) == 1, (rank, category, files)
        actual_ids.append((f"actual_r{rank}_{category}", torch.from_numpy(np.load(files[0]))))
    cases = [(f"synthetic_{n}", metadata(n)) for n in (24576, 24577, 27988, 28608, 28609, 28671, 28672, 28673, 32680, 32704, 32705, 32767, 32768)]
    cases += actual_ids
    for label, cpu_ids in cases:
        persist(label)
        n = len(cpu_ids)
        cpu_counts, cpu_slots = counts_and_slots(cpu_ids)
        ids = cpu_ids.to("musa")
        hidden = torch.randn(n, 2048, device="musa", dtype=torch.bfloat16)
        p = torch.rand(n, 8, device="musa", dtype=torch.float32)
        dy = torch.randn(len(cpu_slots), 2048, device="musa", dtype=torch.bfloat16)
        dp = torch.randn(len(cpu_slots), device="musa", dtype=torch.float32)
        control_namespace["_gate_cpu_counts"] = cpu_counts
        outputs = []
        start = dict(trial.COUNTERS)
        for function in (control, candidate, control):
            x, pp = hidden.clone().requires_grad_(), p.clone().requires_grad_()
            result = function(x, ids, pp, 32, cpu_counts.tolist())
            assert result is not None, (label, reference.last_fallback_reason)
            y, pr, _, got_counts, row_map, _ = result
            dx, dpr = torch.autograd.grad((y, pr), (x, pp), (dy, dp))
            outputs.append([y.detach(), pr.detach(), got_counts, row_map, dx, dpr])
        for name, before, got, after in zip(("FW_hidden", "FW_probability", "counts", "map", "DX", "DP"), *outputs):
            exact(before, after, f"{name}_intended_native_repeat")
            exact(got, before, f"{name}_candidate_vs_intended_TE")
        exact(outputs[1][2], cpu_counts, "counts_vs_CPU")
        delta = {k: v - start.get(k, 0) for k, v in trial.COUNTERS.items() if v != start.get(k, 0)}
        expected = "Long_input_column_sum" if (24577 <= n <= 28608 or 28673 <= n <= 32704) else "original_Bool_column_sum"
        assert delta == {expected: 1}, delta
        # Raw FP64 sums for 128 rows, directly from original incoming BF16 dy.
        selected = cpu_slots // 8 < 128
        gold = torch.zeros(128, 2048, dtype=torch.float64)
        gold.index_add_(0, (cpu_slots // 8)[selected], dy.cpu()[selected].double())
        dx_metric = rms(outputs[1][4][:128], gold)
        assert dx_metric <= 0.01, dx_metric
        exact(outputs[1][4][:128].cpu(), gold.bfloat16(), "DX_CPU_FP64_rounded")
        gold_dp = torch.zeros(n * 8, dtype=torch.float32)
        gold_dp[cpu_slots] = dp.cpu()
        exact(outputs[1][5], gold_dp.reshape(n, 8), "DP_CPU_scatter")
        REPORT["cases"].append({"label": label, "tokens": n, "counter_delta": delta,
                                "DX_raw_FP64_RMS_128_rows": dx_metric})
        del outputs, hidden, p, dy, dp, ids, x, pp, result, y, pr, dx, dpr
    # Weighted merging includes the original BF16 probability cast/product rounding.
    # CPU oracle is restricted to 128 rows; all native/candidate tensors compare raw bits.
    for n in (27988, 32680):
        persist(f"weighted_merge_checkpoint_{n}")
        cpu_ids = metadata(n)
        cpu_counts, cpu_slots = counts_and_slots(cpu_ids)
        ids = cpu_ids.to("musa")
        hidden = torch.randn(n, 2048, device="musa", dtype=torch.bfloat16)
        probs = torch.rand(n, 8, device="musa")
        incoming = torch.randn(n, 2048, device="musa", dtype=torch.bfloat16)
        control_namespace["_gate_cpu_counts"] = cpu_counts
        runs = []
        for function in (control, candidate, control):
            x, p = hidden.clone().requires_grad_(), probs.clone().requires_grad_()
            def merged(xx, pp):
                out = function(xx, ids, pp, 32, cpu_counts.tolist())
                assert out is not None
                weighted = out[0] * out[1].to(out[0].dtype).unsqueeze(-1)
                return reference._TECompactUnpermute.apply(weighted, out[4], 32, n)
            from torch.utils.checkpoint import checkpoint
            output = checkpoint(merged, x, p, use_reentrant=False, determinism_check="default", preserve_rng_state=True)
            dx, dp = torch.autograd.grad(output, (x, p), incoming)
            runs.append([output.detach(), dx, dp])
        for label, a, b, c in zip(("weighted_FW", "weighted_DX", "weighted_DP"), *runs):
            exact(a, c, label + "_intended_native_repeat")
            exact(b, a, label + "_candidate_vs_intended_TE")
        cpu_x, cpu_p, cpu_dy = hidden[:128].cpu(), probs[:128].cpu(), incoming[:128].cpu()
        gold_fw = torch.zeros(128, 2048, dtype=torch.float64)
        gold_dx = torch.zeros_like(gold_fw)
        gold_dp = torch.zeros(128, 8, dtype=torch.float64)
        for slot in range(8):
            valid = cpu_ids[:128, slot] >= 0
            p_bf = cpu_p[:, slot].bfloat16()
            fw_product = (cpu_x.double() * p_bf.double()[:, None]).bfloat16().double()
            dx_product = (cpu_dy.double() * p_bf.double()[:, None]).bfloat16().double()
            gold_fw[valid] += fw_product[valid]
            gold_dx[valid] += dx_product[valid]
            # Native multiplication backward rounds each BF16 product before its sum.
            dp_product = (cpu_dy.double() * cpu_x.double()).bfloat16().double()
            gold_dp[valid, slot] = dp_product[valid].sum(-1).bfloat16().double()
        metrics = {label: rms(value[:128], gold) for label, value, gold in
                   (("FW", runs[1][0], gold_fw), ("DX", runs[1][1], gold_dx), ("DP", runs[1][2], gold_dp))}
        assert all(value <= .01 for value in metrics.values()), metrics
        exact(runs[1][0][:128], gold_fw.bfloat16(), "weighted_FW_CPU_rounded")
        exact(runs[1][1][:128], gold_dx.bfloat16(), "weighted_DX_CPU_rounded")
        exact(runs[1][2][:128], gold_dp.float(), "weighted_DP_CPU_rounded")
        REPORT.setdefault("weighted_checkpoint_cases", []).append({"tokens": n, "raw_FP64_RMS_128_rows": metrics,
                                                                   "nonreentrant_default_checkpoint_rawbits": True})
        del runs, hidden, probs, incoming, x, p, output, dx, dp, ids
    # Genuine duplicate, mismatched authoritative counts, all-invalid routes.
    for kind in ("duplicate", "expert_mismatch"):
        persist(f"validation_{kind}")
        ids_cpu = metadata(27988)
        if kind == "duplicate":
            ids_cpu[1, 0] = ids_cpu[1, 1] = 4
        counts, _ = counts_and_slots(ids_cpu)
        if kind == "expert_mismatch":
            counts[0] += 1
        h = torch.zeros(27988, 2048, device="musa", dtype=torch.bfloat16)
        p = torch.zeros(27988, 8, device="musa")
        result = candidate(h, ids_cpu.to("musa"), p, 32, counts.tolist())
        assert result is None
        assert reference.last_fallback_reason == ("duplicate_expert_slots" if kind == "duplicate" else "expert_counts_mismatch")
        REPORT.setdefault("validation", []).append({"kind": kind, "preserved": True})
        del h, p, result
    # Real original fallback and integer-workaround TE: 2 FW plus 1 BW, inclusive wall.
    for n in (27988, 32680, 32768):
        persist(f"benchmark_{n}")
        ids_cpu = metadata(n)
        counts, slots = counts_and_slots(ids_cpu)
        ids = ids_cpu.to("musa")
        hidden = torch.randn(n, 2048, device="musa", dtype=torch.bfloat16, requires_grad=True)
        p = torch.rand(n, 8, device="musa", requires_grad=True)
        dy = torch.randn(len(slots), 2048, device="musa", dtype=torch.bfloat16)
        dp = torch.randn(len(slots), device="musa")
        def original_dispatch(x, ii, pp, e, cc):
            out = original(x, ii, pp, e, cc)
            return native_fallback(x, ii, pp, e, cc) if out is None else out
        def candidate_dispatch(x, ii, pp, e, cc):
            out = candidate(x, ii, pp, e, cc)
            return native_fallback(x, ii, pp, e, cc) if out is None else out
        def complete(function):
            with torch.no_grad():
                first = function(hidden, ids, p, 32, counts.tolist())
            del first
            out = function(hidden, ids, p, 32, counts.tolist())
            gradients = torch.autograd.grad((out[0], out[1]), (hidden, p), (dy, dp))
            del out, gradients
        measurements = []
        for label, function in (("actual_native_before", original_dispatch), ("candidate", candidate_dispatch), ("actual_native_after", original_dispatch)):
            for _ in range(3):
                complete(function)
            torch.musa.synchronize()
            torch.musa.reset_peak_memory_stats()
            samples = []
            for _ in range(3):
                begin = time.perf_counter_ns()
                for _ in range(8):
                    complete(function)
                torch.musa.synchronize()
                samples.append((time.perf_counter_ns() - begin) / 8e6)
            measurements.append({"variant": label, "wall_ms": samples, "median_ms": statistics.median(samples),
                                 "peak_allocated_bytes": torch.musa.max_memory_allocated(),
                                 "peak_reserved_bytes": torch.musa.max_memory_reserved()})
        REPORT["benchmarks"].append({"tokens": n, "measurements": measurements,
                                     "scope": "Actual original fallback baseline, not intended TE CPU-count numeric control. Complete two FW/one BW includes count cast, allocations, maps/checks/copies/sync."})
        del hidden, ids, p, dy, dp
    persist("outer_zero_assignments_fallback")
    install_te_compact.install()
    start = dict(trial.COUNTERS)
    for n in (0, 27988):
        h = torch.zeros(n, 2048, device="musa", dtype=torch.bfloat16)
        ids = torch.full((n, 8), -1, device="musa", dtype=torch.long)
        p = torch.zeros(n, 8, device="musa")
        out = deepep_ace._compact_permute(h, ids, p, 32, [0] * 32)
        assert out[0].shape == (0, 2048)
    assert dict(trial.COUNTERS) == start
    REPORT["outer_zero_assignments_original_fallback"] = True
    for name, digest in frozen.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    REPORT.update(status="passed_operator_gates", trial_stats=trial.stats())
    persist("completed")


try:
    main()
except BaseException:
    REPORT.update(status="failed", error=traceback.format_exc(), trial_stats=trial.stats())
    persist(REPORT.get("stage", "initialization"))
    raise
