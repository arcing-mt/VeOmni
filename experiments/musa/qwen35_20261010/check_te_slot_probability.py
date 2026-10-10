"""Exact probability-copy operator checks and complete compactor timings."""
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
import torch_musa
import install_te_compact
import install_te_slot_probability
import te_compact_prototype as reference
import te_slot_probability

PROGRESS = {"status": "running", "stage": "initializing", "cases": []}


def checkpoint(stage):
    PROGRESS["stage"] = stage
    Path(os.environ["TE_SLOT_GATE_REPORT"]).write_text(json.dumps(PROGRESS, indent=2) + "\n")


def metadata(tokens, width=32, top_k=8):
    indices = (torch.arange(tokens)[:, None] + torch.arange(top_k)[None, :] * 3) % width
    indices[(torch.arange(tokens)[:, None] + torch.arange(top_k)[None, :] * 2) % 5 == 0] = -1
    indices[::17] = -1
    valid = torch.nonzero(indices.flatten() >= 0).flatten()
    slots = valid[torch.argsort(indices.flatten()[valid], stable=True)]
    counts = torch.bincount(indices[indices >= 0], minlength=width).tolist()
    return indices, slots, counts


def exact(actual, expected, label):
    # Check signed zeros and NaN payloads as copies, too.
    assert actual.dtype == expected.dtype and actual.shape == expected.shape, label
    dtype = torch.int32 if actual.dtype == torch.float32 else torch.int16
    if actual.is_floating_point():
        assert torch.equal(actual.detach().contiguous().view(dtype), expected.detach().contiguous().view(dtype)), label
    else:
        assert torch.equal(actual, expected), label


def raw_rms(actual, raw_reference):
    difference = actual.detach().cpu().double() - raw_reference
    return float((difference.square().sum() / raw_reference.square().sum().clamp_min(1e-30)).sqrt())


def compare_case(original, candidate, tokens, width=32, hidden_width=2048, mutate_metadata=False, strided_gradient=False, probability_only=False, special=False, special_gradient=False):
    indices_cpu, slots, counts = metadata(tokens, width)
    hidden = torch.randn(tokens, hidden_width, device="musa", dtype=torch.bfloat16)
    probabilities = torch.rand(tokens, 8, device="musa", dtype=torch.float32)
    if special:
        probabilities[:, 0] = -0.0
        probabilities[:, 1] = float("inf")
        probabilities[:, 2] = float("nan")
    routes = indices_cpu.to("musa")
    outputs, gradients = [], []
    counters_before = dict(te_slot_probability.COUNTERS)
    dy = torch.randn(len(slots), hidden_width * (2 if strided_gradient else 1), device="musa", dtype=torch.bfloat16)
    dp = torch.randn(len(slots) * (2 if strided_gradient else 1), device="musa")
    if strided_gradient:
        dy, dp = dy[:, ::2], dp[::2]
    if special_gradient:
        assert not strided_gradient
        bits = [0, 0x80000000, 1, 0x80000001, 0x00800000, 0x007fffff, 0x7f800000, 0xff800000,
                0x7f800001, 0x7fc12345, 0xff800001, 0x3f800000]
        signed = [v if v < 2**31 else v - 2**32 for v in bits]
        dp = torch.tensor(signed, dtype=torch.int32, device="musa").repeat(triton_cdiv(len(slots), len(signed)))[:len(slots)].view(torch.float32)
    for variant, function in (("original_before", original), ("candidate", candidate), ("original_after", original)):
        x = hidden.detach().clone().requires_grad_(not probability_only)
        p = probabilities.detach().clone().requires_grad_()
        ids = routes.clone()
        result = function(x, ids, p, width, counts)
        if result is None:
            raise RuntimeError(f"Compactor rejected case: tokens={tokens}, variant={variant}, reason={reference.last_fallback_reason}")
        y, pr, _, got_counts, row_map, _ = result
        if mutate_metadata:
            ids.fill_(-1)
        with torch.no_grad():
            outputs.append((y.detach().clone(), pr.detach().clone(), got_counts.detach().clone()))
        if probability_only:
            gradients.append((torch.autograd.grad(pr, p, dp)[0],))
        else:
            gradients.append(torch.autograd.grad((y, pr), (x, p), (dy, dp)))
        del result, y, pr, row_map, x, p, ids
    for item in range(3):
        exact(outputs[0][item], outputs[1][item], "forward")
        exact(outputs[0][item], outputs[2][item], "native repeat")
    for item in range(len(gradients[0])):
        exact(gradients[0][item], gradients[1][item], "gradient")
        exact(gradients[0][item], gradients[2][item], "native repeat gradient")
    delta = {key: value - counters_before.get(key, 0) for key, value in te_slot_probability.COUNTERS.items()
             if value != counters_before.get(key, 0)}
    expected_delta = {"fused_weights": 1, "direct_slot_gradient": 1} if hidden_width == 2048 else {"native_weights": 1}
    assert delta == expected_delta, (delta, expected_delta)
    raw_metrics = None
    if tokens <= 257 and not probability_only:
        raw_gold_dx = torch.zeros(tokens, hidden_width, dtype=torch.float64).index_add_(0, slots // 8, dy.cpu().double())
        gold_dx = raw_gold_dx.bfloat16()
        exact(gradients[1][0].cpu(), gold_dx, "CPU FP64 dx")
        raw_metrics = {"dx_relative_rms_candidate": raw_rms(gradients[1][0], raw_gold_dx),
                       "dx_relative_rms_original": raw_rms(gradients[0][0], raw_gold_dx),
                       "raw_threshold_fixed_before_GPU": 0.01}
        assert raw_metrics["dx_relative_rms_candidate"] <= 0.01
        assert raw_metrics["dx_relative_rms_original"] <= 0.01
        if not special_gradient:
            gold_dp = torch.zeros(tokens * 8, dtype=torch.float32)
            gold_dp[slots] = dp.cpu()
            exact(gradients[1][1].cpu(), gold_dp.reshape(tokens, 8), "CPU dp")
            raw_metrics["dp_relative_rms_candidate"] = raw_rms(gradients[1][1], gold_dp.reshape(tokens, 8).double())
            assert raw_metrics["dp_relative_rms_candidate"] == 0
        exact(outputs[1][0].cpu(), hidden.cpu()[slots // 8], "CPU forward hidden")
        exact(outputs[1][1].cpu(), probabilities.cpu().flatten()[slots], "CPU forward probabilities")
    return {"tokens": tokens, "experts": width, "hidden_width": hidden_width,
            "mutate_metadata": mutate_metadata, "strided_gradient": strided_gradient,
            "probability_only": probability_only, "special_probabilities": special,
            "special_gradients": special_gradient,
            "original_candidate_original_FW_DX_DP_bitwise": True,
            "counter_delta": delta, "raw_FP64_metrics": raw_metrics,
            "CPU_FP64_smallcase": tokens <= 257 and not probability_only}


def triton_cdiv(a, b):
    return (a + b - 1) // b


def native_rejected_shape_case(original, candidate, tokens):
    # This installed backend's bool column reduction is wrong at these N.
    # Both full helpers must preserve the original rejection before new code.
    ids_cpu, _, counts = metadata(tokens)
    hidden = torch.zeros(tokens, 2048, device="musa", dtype=torch.bfloat16)
    probabilities = torch.ones(tokens, 8, device="musa")
    indices = ids_cpu.to("musa")
    before = dict(te_slot_probability.COUNTERS)
    reasons = []
    for function in (original, candidate, original):
        assert function(hidden, indices, probabilities, 32, counts) is None
        reasons.append(reference.last_fallback_reason)
    assert reasons == ["duplicate_expert_slots"] * 3, reasons
    assert dict(te_slot_probability.COUNTERS) == before
    return {"tokens": tokens, "original_candidate_original_rejection_exact": True,
            "reasons": reasons, "candidate_kernel_not_executed": True,
            "scope": "Existing native bool-column sum failure retained. This shape is not a probability kernel numerical pass."}


def checkpoint_case(original, candidate):
    from torch.utils.checkpoint import checkpoint as torch_checkpoint
    ids, slots, counts = metadata(257)
    ids = ids.to("musa")
    hidden = torch.randn(257, 2048, device="musa", dtype=torch.bfloat16)
    probabilities = torch.rand(257, 8, device="musa")
    dy = torch.randn(len(slots), 2048, device="musa", dtype=torch.bfloat16)
    dp = torch.randn(len(slots), device="musa")
    results = []
    before = dict(te_slot_probability.COUNTERS)
    for function in (original, candidate, original):
        x = hidden.clone().requires_grad_()
        p = probabilities.clone().requires_grad_()

        def forward(xx, pp):
            result = function(xx, ids, pp, 32, counts)
            return result[0], result[1]

        y, pr = torch_checkpoint(forward, x, p, use_reentrant=False, determinism_check="default", preserve_rng_state=True)
        results.append((y.detach().clone(), pr.detach().clone(), *torch.autograd.grad((y, pr), (x, p), (dy, dp))))
    for index in range(4):
        exact(results[0][index], results[1][index], "checkpoint exact")
        exact(results[0][index], results[2][index], "checkpoint original repeat")
    delta = {key: value - before.get(key, 0) for key, value in te_slot_probability.COUNTERS.items() if value != before.get(key, 0)}
    assert delta == {"fused_weights": 2, "direct_slot_gradient": 1}, delta
    return {"nonreentrant_default_metadata": True, "FW_DX_DP_exact": True, "counter_delta": delta}


def pending_input_case(original):
    ids_cpu, slots, counts = metadata(257)
    hidden = torch.randn(257, 2048, device="musa", dtype=torch.bfloat16)
    ids = ids_cpu.to("musa")
    probabilities = torch.rand(257, 8, device="musa")
    with torch.no_grad():
        baseline = original(hidden, ids, probabilities, 32, counts)
        row_map = baseline[4]
        row_nt = row_map.t().contiguous()
    torch.musa.synchronize()
    gold_y, gold_p = baseline[0].cpu(), baseline[1].cpu()
    producer, consumer = torch.musa.Stream(), torch.musa.Stream()
    ready = torch.musa.Event()
    a = torch.ones(4096, 4096, device="musa")
    b = torch.ones_like(a)
    torch.musa.synchronize()
    with torch.musa.stream(producer):
        for _ in range(32):
            scratch = a @ b
        produced_ids = ids.clone()
        produced_probabilities = probabilities.clone()
        ready.record(producer)
    pending_before = not ready.query()
    assert pending_before, "Producer completed; pending-input proof was not exercised"
    with torch.no_grad(), torch.musa.stream(consumer):
        consumer.wait_event(ready)
        # Bypass already-tested counts to isolate new pointer lifetime reads.
        result = te_slot_probability._TECompactSlotProbabilities.apply(hidden, produced_probabilities, produced_ids, row_nt, row_map, 257, 32, len(slots))
        snapshot_y, snapshot_p = result[0].clone(), result[1].clone()
    del produced_ids, produced_probabilities, result
    # Attempt same-size producer-allocator reuse after dropping all new inputs.
    with torch.musa.stream(producer):
        for _ in range(32):
            replacement_ids = torch.full_like(ids, -1)
            replacement_probabilities = torch.full_like(probabilities, -777)
    torch.musa.synchronize()
    exact(snapshot_y.cpu(), gold_y, "pending hidden")
    exact(snapshot_p.cpu(), gold_p, "pending probabilities")
    return {"producer_ready_pending_before_candidate": pending_before,
            "explicit_consumer_event_dependency": True, "dropped_new_input_refs_and_reuse_stress": True,
            "exact": True, "scope": "Isolated new FW probability/index reads. Original native hidden input and maps remain alive; not a proof for every native extension or backward branch."}


def main():
    names = ("check_te_slot_probability.py", "te_slot_probability.py", "install_te_slot_probability.py",
             "te_compact_prototype.py", "install_te_compact.py")
    source_sha = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}
    PROGRESS["source_sha256"] = source_sha
    os.environ["TE_COMPACT"] = "1"
    os.environ["TE_SLOT_PROBABILITY"] = "1"
    torch.musa.set_device(0)
    torch.manual_seed(44)
    install_te_compact.load_native()
    original = reference._te_compact_permute
    assert install_te_slot_probability.install()
    candidate = reference._te_compact_permute
    rows = PROGRESS["cases"]
    for tokens in (17, 257, 2048, 8192, 32768, 40960, 41047):
        checkpoint(f"ordinary_{tokens}")
        rows.append(compare_case(original, candidate, tokens))
        checkpoint(f"ordinary_{tokens}_passed")
    checkpoint("original_rejected_shapes")
    rejected_shapes = [native_rejected_shape_case(original, candidate, tokens) for tokens in (32680, 32681)]
    checkpoint("edge_cases")
    rows.append(compare_case(original, candidate, 257, mutate_metadata=True, strided_gradient=True))
    rows.append(compare_case(original, candidate, 257, probability_only=True))
    rows.append(compare_case(original, candidate, 257, special=True))
    rows.append(compare_case(original, candidate, 257, special_gradient=True))
    rows.append(compare_case(original, candidate, 257, hidden_width=128))
    checkpoint("checkpoint_case")
    checkpoint_result = checkpoint_case(original, candidate)
    checkpoint("pending_input_case")
    pending_result = pending_input_case(original)
    checkpoint("duplicate_and_higher_order")
    # Duplicate slots must be rejected before reaching either new kernel.
    ids, _, counts = metadata(257)
    ids[1, 0] = ids[1, 1] = 2
    counts = torch.bincount(ids[ids >= 0], minlength=32).tolist()
    x = torch.ones(257, 2048, device="musa", dtype=torch.bfloat16)
    p = torch.ones(257, 8, device="musa")
    before = dict(te_slot_probability.COUNTERS)
    assert original(x, ids.to("musa"), p, 32, counts) is None
    assert candidate(x, ids.to("musa"), p, 32, counts) is None
    assert dict(te_slot_probability.COUNTERS) == before
    assert reference.last_fallback_reason == "duplicate_expert_slots"
    # Probability-only higher order: original gather graph vs new fallback.
    ids, slots, counts = metadata(17)
    second = []
    for function in (original, candidate):
        pp = torch.ones(17, 8, device="musa", requires_grad=True)
        result = function(torch.ones(17, 2048, device="musa", dtype=torch.bfloat16), ids.to("musa"), pp, 32, counts)
        first = torch.autograd.grad(result[1].square().sum(), pp, create_graph=True)[0]
        second.append(torch.autograd.grad(first.sum(), pp)[0])
    exact(second[0], second[1], "higher order probability")
    expected = torch.zeros(17 * 8, device="musa")
    expected[slots.to("musa")] = 2
    exact(second[1], expected.reshape(17, 8), "higher order oracle")
    # No-grad behavior uses the same original helper outputs.
    with torch.no_grad():
        a = original(x, metadata(257)[0].to("musa"), p, 32, metadata(257)[2])
        b = candidate(x, metadata(257)[0].to("musa"), p, 32, metadata(257)[2])
        exact(a[0], b[0], "no grad hidden")
        exact(a[1], b[1], "no grad probabilities")
    result = {"status": "passed_operator_gates", "cases": rows, "duplicate_rejection_unchanged": True,
              "source_sha256": source_sha,
              "native_rejected_shapes": rejected_shapes,
              "checkpoint": checkpoint_result, "pending_input": pending_result,
              "probability_higher_order_exact": True, "no_grad_exact": True,
              "counters": install_te_slot_probability.stats(),
              "scope": "Pure compactor gates only; no FSDP/Adam/fullmodel/performance acceptance. Original native hidden permutation/adjoint retained. NaN copy payloads, metadata reuse and probability-only/strided gradients included. Numerical thresholds not relaxed."}
    assert all(hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == digest for name, digest in source_sha.items())
    Path(os.environ["TE_SLOT_GATE_REPORT"]).write_text(json.dumps(result, indent=2) + "\n")
    PROGRESS.update(status="passed_operator_gates")
    print(json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        PROGRESS.update(status="failed", error=f"{type(error).__name__}: {error}")
        try:
            checkpoint(PROGRESS["stage"])
        except Exception as output_error:
            print(f"Partial report failed: {output_error}", file=sys.stderr)
        raise
