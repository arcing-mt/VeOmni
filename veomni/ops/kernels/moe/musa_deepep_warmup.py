"""Initialize the unchanged native fallback with isolated synthetic tensors."""

import os
import time

import torch


def warm_native_fallback(device, native_permute):
    if device.type != "musa" or device.index != torch.musa.current_device():
        raise RuntimeError("Native warmup requires the trainer's current MUSA device")
    if os.environ.get("VEOMNI_MUSA_DEEPEP_COUNTING_SORT", "0").lower() not in {"1", "true", "yes", "on"}:
        raise RuntimeError("Native warmup requires the unchanged counting-sort fallback")
    result = {
        "device": str(device),
        "calls": [],
        "status": "running",
        "scope": "Synthetic native calls only, no model/data/RNG/optimizer/collective. No added device sync. All initialization CPU time remains in measured training step1; original 50 updates and steps10-50 remain intact.",
    }
    started = time.perf_counter_ns()
    with torch.no_grad():
        # num_slots: divisible16 yes/no; num_blocks: divisible16 yes/no,
        # positive >1. These four small inputs cover the installed Triton
        # integer specialization classes used by the observed real payloads.
        for tokens in (256, 257, 4096, 4095):
            indices_cpu = torch.full((tokens, 8), -1, dtype=torch.long)
            indices_cpu[:, 0] = torch.arange(tokens) % 32
            indices_cpu[0, 1] = 0
            counts = torch.bincount(indices_cpu[indices_cpu >= 0], minlength=32).tolist()
            indices = indices_cpu.to(device)
            hidden = torch.zeros((tokens, 2048), device=device, dtype=torch.bfloat16)
            probabilities = torch.ones((tokens, 8), device=device, dtype=torch.float32)
            begin = time.perf_counter_ns()
            outputs = native_permute(hidden, indices, probabilities, 32, counts)
            result["calls"].append(
                {
                    "tokens": tokens,
                    "num_slots": tokens * 8,
                    "num_blocks": (tokens * 8 + 1023) // 1024,
                    "cpu_ms": (time.perf_counter_ns() - begin) / 1e6,
                }
            )
            del outputs, indices, hidden, probabilities
    result.update(status="passed", total_cpu_ms=(time.perf_counter_ns() - started) / 1e6)
    return result


_WARMED_DEVICES = set()


def maybe_warm_native_fallback(device, native_permute):
    """One-shot synthetic initialization inside the first real compaction call."""
    if os.environ.get("VEOMNI_MUSA_ACE_NATIVE_WARMUP", "0") != "1" or device.type != "musa":
        return
    if device not in _WARMED_DEVICES:
        warm_native_fallback(device, native_permute)
        _WARMED_DEVICES.add(device)
