"""Exact native FW/DX/rstd and raw FP64 RMS<1%; complete preentry FW/BW timing."""

import hashlib
import json
from pathlib import Path
import statistics
import time
import traceback

import torch
import torch_musa

import qk_repeat_l2norm as prototype


root = Path(__file__).parent
report = {
    "status": "running", "results": [], "timings": [], "stream_cases": [], "fallbacks": [],
    "criteria": {"FW_DX_rstd_exact": True, "native_repeat_exact": True, "raw_FP64_RMS": .01},
    "source_sha256": {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in
                      ("qk_repeat_l2norm.py", "qk_repeat_l2norm_kernel.py", "check_qk_repeat_l2norm.py")},
}


def persist():
    (root / "qk_repeat_l2norm_gate.json").write_text(json.dumps(report, indent=2) + "\n")


def compare(a, b):
    assert a.shape == b.shape
    a, b = a.detach().cpu().double(), b.detach().cpu().double()
    return {"equal": torch.equal(a, b), "max_abs": (a - b).abs().max().item(),
            "rms": ((a - b).square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-12)).item()}


def operation(kind, packed, dys, debug=False):
    packed = packed.detach().requires_grad_()
    q, k, _ = packed.split((2048, 2048, 4096), dim=-1)
    q, k = (x.reshape(1, packed.shape[1], 16, 128) for x in (q, k))
    fn = prototype.native if kind == "native" else prototype.repeat_norm
    yq, yk = fn(q), fn(k)
    auxiliary = (yq.grad_fn.saved_tensors[1].detach(), yk.grad_fn.saved_tensors[1].detach()) if debug else ()
    dx, = torch.autograd.grad((yq, yk), packed, dys)
    return yq.detach(), yk.detach(), dx, *auxiliary


def selected():
    return {n: str(getattr(prototype._native, n).best_config) for n in
            ("l2norm_fwd_kernel", "l2norm_bwd_kernel")}


def timing(kind, packed, dys):
    for _ in range(3):
        operation(kind, packed, dys)
    torch.musa.synchronize()
    torch.musa.reset_peak_memory_stats()
    allocated_before = torch.musa.memory_allocated()
    samples = []
    for _ in range(16):
        start = time.perf_counter()
        operation(kind, packed, dys)
        torch.musa.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return {"samples_ms": samples, "mean_ms": statistics.mean(samples),
            "median_ms": statistics.median(samples),
            "extra_peak_bytes": torch.musa.max_memory_allocated() - allocated_before}


def oracle(packed, dys):
    x = packed.detach().cpu().double().requires_grad_()
    q, k, _ = x.split((2048, 2048, 4096), dim=-1)
    outputs = []
    for t in (q, k):
        t = t.reshape(1, x.shape[1], 16, 128).repeat_interleave(2, dim=2)
        outputs.append(t / (t.square().sum(-1, keepdim=True) + 1e-6).sqrt())
    dx, = torch.autograd.grad(outputs, x, [d.detach().cpu().double() for d in dys])
    return *outputs, dx


try:
    torch.musa.set_device(0)
    torch.manual_seed(40951)
    prototype.initialize()
    # Exercise a naturally cold native autotuner without clearing or forcing it.
    assert not prototype._native.l2norm_fwd_kernel.cache
    packed = torch.randn(1, 17, 8192, device="musa", dtype=torch.bfloat16)
    x = packed[..., :2048].reshape(1, 17, 16, 128).detach().requires_grad_()
    assert prototype.eligible(x, 1e-6) and prototype.native_config(x) is None
    dy = torch.randn(1, 17, 32, 128, device="musa", dtype=torch.bfloat16)
    c = prototype.repeat_norm(x)
    cdx, = torch.autograd.grad(c, x, dy)
    assert prototype.stats["native_config_fallback"] == 1 and prototype.stats["fused_forward"] == 0
    n = prototype.native(x)
    ndx, = torch.autograd.grad(n, x, dy)
    cold = {"case": "eligible_cold_native_config", "errors": [compare(c, n), compare(cdx, ndx)],
            "native_config_fallback": 1, "candidate_launches": 0, "native_cache_key":
            [list(key) for key in prototype._native.l2norm_fwd_kernel.cache]}
    report["fallbacks"].append(cold)
    persist()
    assert all(e["equal"] for e in cold["errors"]), cold
    del packed, x, dy, c, cdx, n, ndx
    # Keep the original regression inputs despite the added cold-key case.
    torch.manual_seed(40951)
    for seq in (1, 17, 196, 2048, 8155, 8192):
        packed = torch.randn(1, seq, 8192, dtype=torch.bfloat16, device="musa")
        dys = tuple(torch.randn(1, seq, 32, 128, dtype=torch.bfloat16, device="musa") * .1 for _ in range(2))
        snapshots = [t.cpu().clone() for t in (packed, *dys)]
        versions = [t._version for t in (packed, *dys)]
        expected = operation("native", packed, dys, debug=True)
        repeated = operation("native", packed, dys, debug=True)
        hits = prototype.stats.copy()
        actual = operation("candidate", packed, dys, debug=True)
        probe = packed[..., :2048].reshape(1, seq, 16, 128)
        supported = prototype.eligible(probe, 1e-6)
        if seq == 1:
            # Reshape canonicalizes this singleton token stride to 2048.
            # Preserve the strict prototype guard and test its native fallback.
            assert not supported and probe.stride(1) == 2048
            assert prototype.stats["fallback"] == hits["fallback"] + 2
            assert prototype.stats["fused_forward"] == hits["fused_forward"]
            assert prototype.stats["fused_backward"] == hits["fused_backward"]
        else:
            assert supported
            assert prototype.stats["fused_forward"] == hits["fused_forward"] + 2
            assert prototype.stats["fused_backward"] == hits["fused_backward"] + 2
            assert str(prototype.native_config(probe)) == selected()["l2norm_fwd_kernel"]
        errors = [compare(a, b) for a, b in zip(actual, expected)]
        repeats = [compare(a, b) for a, b in zip(repeated, expected)]
        row = {"seq": seq, "qk_stride": list(probe.stride()), "supported": supported,
               "errors": errors, "native_repeat": repeats, "native_selected_configs": selected()}
        report["results"].append(row)
        persist()
        assert all(e["equal"] for e in errors + repeats), row
        assert versions == [t._version for t in (packed, *dys)]
        assert all(torch.equal(t.cpu(), s) for t, s in zip((packed, *dys), snapshots))
        if seq == 196:
            gold = oracle(packed, dys)
            row["raw_FP64_errors"] = [compare(a, b) for a, b in zip(actual[:3], gold)]
            persist()
            assert all(e["rms"] < .01 for e in row["raw_FP64_errors"]), row
            for scale in (0., 1e-5):
                tiny = packed * scale
                n, c = operation("native", tiny, dys, debug=True), operation("candidate", tiny, dys, debug=True)
                extra = {"seq": seq, "scale": scale, "errors": [compare(a, b) for a, b in zip(c, n)]}
                report["results"].append(extra)
                assert all(e["equal"] for e in extra["errors"]), extra
        if seq >= 196:
            report["timings"].append({"seq": seq, "native_before": timing("native", packed, dys),
                                      "candidate": timing("candidate", packed, dys),
                                      "native_after": timing("native", packed, dys)})
            persist()
        if seq == 2048:
            gold = [t.cpu().clone() for t in expected]
            # Noncontiguous upstream gradients exercise the preserved guard copy.
            strided_dys = tuple(torch.empty(1, seq, 32, 256, device="musa", dtype=torch.bfloat16)[..., ::2]
                                for _ in range(2))
            for t, src in zip(strided_dys, dys):
                t.copy_(src)
            n = operation("native", packed, strided_dys, debug=True)
            c = operation("candidate", packed, strided_dys, debug=True)
            row["strided_dy_errors"] = [compare(a, b) for a, b in zip(c, n)]
            assert all(e["equal"] for e in row["strided_dy_errors"])
            matrix = torch.randn(4096, 4096, device="musa", dtype=torch.bfloat16)
            for kind in ("native", "candidate"):
                producer, consumer = torch.musa.Stream(), torch.musa.Stream()
                producer.wait_stream(torch.musa.current_stream())
                with torch.musa.stream(producer):
                    for _ in range(24):
                        delayed = torch.mm(matrix, matrix)
                    px, pdy = packed.clone(), tuple(t.clone() for t in dys)
                    ready = torch.musa.Event()
                    ready.record()
                pending = not ready.query()
                item = {"kind": kind, "producer_pending": pending}
                report["stream_cases"].append(item)
                persist()
                assert pending
                with torch.musa.stream(consumer):
                    consumer.wait_event(ready)
                    values = [t.clone() for t in operation(kind, px, pdy, debug=True)]
                torch.musa.synchronize()
                item["errors"] = [compare(t.cpu(), g) for t, g in zip(values, gold)]
                persist()
                assert all(e["equal"] for e in item["errors"]), item
    for label, heads, dtype, contiguous, eps, no_grad in (
        ("contiguous", 16, torch.bfloat16, True, 1e-6, False),
        ("heads8", 8, torch.bfloat16, False, 1e-6, False),
        ("float32", 16, torch.float32, False, 1e-6, False),
        ("eps", 16, torch.bfloat16, False, 1e-5, False),
        ("no_grad", 16, torch.bfloat16, False, 1e-6, True),
    ):
        packed = torch.randn(1, 17, 8192, dtype=dtype, device="musa")
        x = packed[..., :heads * 128].reshape(1, 17, heads, 128)
        if contiguous:
            x = x.contiguous()
        x = x.detach().requires_grad_()
        before = prototype.stats["fallback"]
        with torch.set_grad_enabled(not no_grad):
            n, c = prototype.native(x, eps), prototype.repeat_norm(x, eps)
            errors = [compare(c, n)]
            if not no_grad:
                dy = torch.randn_like(n)
                ndx, = torch.autograd.grad(n, x, dy)
                cdx, = torch.autograd.grad(c, x, dy)
                errors.append(compare(cdx, ndx))
        item = {"case": label, "errors": errors, "fallback_delta": prototype.stats["fallback"] - before}
        report["fallbacks"].append(item)
        persist()
        assert item["fallback_delta"] == 1 and all(e["equal"] for e in errors), item
    report.update(status="passed", stats=dict(prototype.stats),
                  scope="Isolated q/k preentry helper; original actual cached FW config/native BW. Cold config uses complete native fallback. Exact operator only; no model/OpSlot/FSDP/FA/free-trajectory certification or full-step gain.")
    persist()
    print(json.dumps(report, indent=2), flush=True)
except BaseException:
    report.update(status="failed", error=traceback.format_exc(), stats=dict(prototype.stats))
    persist()
    raise
