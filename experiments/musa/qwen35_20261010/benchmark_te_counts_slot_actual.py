"""Actual routing complete permute/weighted merge: two FW and one BW."""
import hashlib
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
import install_te_native_counts as counts_trial
import install_te_counts_slot as trial
import te_slot_probability as slot_trial
import te_compact_prototype as reference

ROOT = Path(__file__).resolve().parent
DESTINATION = Path(os.environ["TE_COUNTS_SLOT_ACTUAL_BENCH_REPORT"])
REPORT = {"status": "running", "cases": []}


def persist(stage):
    REPORT["stage"] = stage
    DESTINATION.write_text(json.dumps(REPORT, indent=2) + "\n")
    print(stage, flush=True)


def main():
    gate_path = Path(os.environ["TE_COUNTS_SLOT_GATE_REPORT"])
    gate = json.loads(gate_path.read_text())
    assert gate["status"] == "passed_operator_gates"
    for name, digest in gate["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    assert os.environ["VEOMNI_MUSA_DEEPEP_COUNTING_SORT"] == "1"
    assert not torch.are_deterministic_algorithms_enabled()
    REPORT.update(gate_sha256=hashlib.sha256(gate_path.read_bytes()).hexdigest(),
                  source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                 for name in (*gate["source_sha256"], Path(__file__).name)},
                  scope="Actual observed routing IDs only; synthetic hidden/probability/dy. Complete "
                        "permute plus BF16 probability cast/product plus correct paired weighted merge. "
                        "Counts-only control before/after. Two FW/one BW inclusive wall; no real expert MLP/ACE/FSDP or train gain claim.",
                  native_counting_sort=1, deterministic_algorithms=False)
    torch.musa.set_device(0)
    torch.manual_seed(4408)
    native_permute, native_unpermute = deepep_ace._compact_permute, deepep_ace._compact_unpermute
    install_te_compact.load_native()
    os.environ["TE_COMPACT"] = "1"
    os.environ["TE_NATIVE_LONG_COUNTS"] = "1"
    assert counts_trial.install()
    original = reference._te_compact_permute
    os.environ["TE_SLOT_PROBABILITY"] = "1"
    os.environ["TE_COUNTS_SLOT_COMBINED"] = "1"
    assert trial.install()
    candidate = reference._te_compact_permute
    raw = Path(os.environ["TE_COUNTS_ACTUAL_RAW"])
    expected = json.loads((ROOT / "te_native_counts_operator_actual_sha256.json").read_text())
    for name, digest in expected.items():
        assert hashlib.sha256((raw / name).read_bytes()).hexdigest() == digest, name
    REPORT["timing_control"] = "Qualified counts-only original TE before/after; slot-copy composition is the sole new variable."
    # All ten real masks span both observed categories; do not select only a winner.
    for path in sorted(raw.glob("te_native_counts_rank*_indices.npy")):
        ids_cpu = torch.from_numpy(np.load(path, allow_pickle=False))
        n = len(ids_cpu)
        counts = torch.bincount(ids_cpu[ids_cpu >= 0], minlength=32).tolist()
        persist(path.name)
        ids = ids_cpu.to("musa")
        hidden = torch.randn(n, 2048, device="musa", dtype=torch.bfloat16, requires_grad=True)
        probabilities = torch.rand(n, 8, device="musa", requires_grad=True)
        incoming = torch.randn(n, 2048, device="musa", dtype=torch.bfloat16)
        path_counts = {"TE": 0, "native_fallback": 0}
        def forward(function):
            out = function(hidden, ids, probabilities, 32, counts)
            if out is None:
                path_counts["native_fallback"] += 1
                y, p, rows, _ = native_permute(hidden, ids, probabilities, 32, counts)
                return native_unpermute(y, p, rows, n)
            path_counts["TE"] += 1
            y, p, _, _, row_map, width = out
            weighted = y * p.to(y.dtype).unsqueeze(-1)
            return reference._TECompactUnpermute.apply(weighted, row_map, width, n)
        def complete(function):
            with torch.no_grad():
                first = forward(function)
            del first
            second = forward(function)
            gradients = torch.autograd.grad(second, (hidden, probabilities), incoming)
            del second, gradients
        measurements = []
        for variant, function in (("counts_only_before", original), ("candidate", candidate), ("counts_only_after", original)):
            for _ in range(3):
                complete(function)
            torch.musa.synchronize()
            torch.musa.reset_peak_memory_stats()
            path_counts.update(TE=0, native_fallback=0)
            counters = dict(trial.COUNTERS)
            slot_counters = dict(slot_trial.COUNTERS)
            samples = []
            for _ in range(3):
                start = time.perf_counter_ns()
                for _ in range(8):
                    complete(function)
                torch.musa.synchronize()
                samples.append((time.perf_counter_ns() - start) / 8e6)
            delta = {k: v - counters.get(k, 0) for k, v in trial.COUNTERS.items() if v != counters.get(k, 0)}
            assert delta == {"Long_input_column_sum": 48}, delta
            slot_delta = {k: v - slot_counters.get(k, 0) for k, v in slot_trial.COUNTERS.items() if v != slot_counters.get(k, 0)}
            assert slot_delta == ({"fused_weights": 48, "direct_slot_gradient": 24} if variant == "candidate" else {}), slot_delta
            expected_paths = {"TE": 48, "native_fallback": 0}
            assert path_counts == expected_paths, path_counts
            measurements.append({"variant": variant, "wall_ms": samples, "median_ms": statistics.median(samples),
                                 "observed_dispatch_paths": dict(path_counts),
                                 "count_trial_delta": delta, "slot_trial_delta": slot_delta, "peak_allocated_bytes": torch.musa.max_memory_allocated(),
                                 "peak_reserved_bytes": torch.musa.max_memory_reserved()})
        REPORT["cases"].append({"source": path.name, "tokens": n, "assignments": sum(counts),
                                "measurements": measurements})
        del hidden, probabilities, incoming, ids
    assert len(REPORT["cases"]) == 10
    for name, digest in REPORT["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    REPORT.update(status="completed_actual_routing_micro", trial_stats=trial.stats())
    persist("completed")


try:
    main()
except BaseException:
    REPORT.update(status="failed", error=traceback.format_exc(), trial_stats=trial.stats())
    persist(REPORT.get("stage", "initialization"))
    raise
