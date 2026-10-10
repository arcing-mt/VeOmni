"""Validate warmup isolation and specialization coverage on the actual wheel."""
import importlib
import json
import os
from pathlib import Path
import time

import torch
import torch_musa
from native_fallback_warmup import native_permute, warm_native_fallback


def main():
    os.environ["VEOMNI_MUSA_DEEPEP_COUNTING_SORT"] = "1"
    torch.musa.set_device(0)
    device = torch.device("musa", 0)
    cpu_rng = torch.get_rng_state().clone()
    musa_rng = torch.musa.get_rng_state().clone()
    grad_enabled = torch.is_grad_enabled()
    deterministic = torch.are_deterministic_algorithms_enabled()
    report = warm_native_fallback(device)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(musa_rng, torch.musa.get_rng_state())
    assert grad_enabled == torch.is_grad_enabled()
    assert deterministic == torch.are_deterministic_algorithms_enabled()
    module = importlib.import_module("veomni.ops.kernels.moe.musa_deepep_compact")
    kernels = [getattr(module, n) for n in ("_count_experts_by_block", "_exclusive_prefix_by_expert", "_scatter_stable_slots")]
    keys_before = [set(k.cache[0]) for k in kernels]
    rows = []
    for tokens in (32680, 32582, 32681, 32583, 40960, 40959):
        indices_cpu = torch.full((tokens, 8), -1, dtype=torch.long)
        indices_cpu[:, 0] = torch.arange(tokens) % 32
        indices_cpu[0, 1] = 0
        counts = torch.bincount(indices_cpu[indices_cpu >= 0], minlength=32).tolist()
        hidden_cpu = ((torch.arange(tokens * 2048) % 31 - 15).reshape(tokens, 2048) / 16).to(torch.bfloat16)
        probabilities_cpu = (torch.arange(tokens * 8).reshape(tokens, 8) % 23).float() / 32
        hidden, indices, probabilities = hidden_cpu.to(device), indices_cpu.to(device), probabilities_cpu.to(device)
        torch.musa.synchronize()
        started = time.perf_counter_ns()
        output = native_permute(hidden, indices, probabilities, 32, counts)
        cpu_ms = (time.perf_counter_ns() - started) / 1e6
        torch.musa.synchronize()
        valid = torch.nonzero(indices_cpu.flatten() >= 0).flatten()
        slots = valid[torch.argsort(indices_cpu.flatten()[valid], stable=True)]
        assert torch.equal(output[0].cpu(), hidden_cpu[slots // 8])
        assert torch.equal(output[1].cpu(), probabilities_cpu.flatten()[slots])
        assert torch.equal(output[2].cpu(), slots // 8)
        assert torch.equal(output[3].cpu(), torch.tensor(counts))
        assert [set(k.cache[0]) for k in kernels] == keys_before, "Observed shape created a new Triton specialization"
        rows.append({"tokens": tokens, "cpu_ms": cpu_ms, "exact": True, "new_triton_specializations": 0})
        del output, hidden, indices, probabilities, hidden_cpu, probabilities_cpu
    result = {"status": "passed", "warmup": report, "rows": rows,
              "rng_and_grad_and_deterministic_unchanged": True,
              "specialization_keys": [sorted(keys) for keys in keys_before],
              "scope": "Original native outputs, isolated warmup only; no new math/gradient implementation. Representative real geometry, duplicates and invalid slots are exact. No model/PG/fulltraining claim. Explicit sync is diagnostic-only."}
    Path(os.environ["WARMUP_GATE_REPORT"]).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
