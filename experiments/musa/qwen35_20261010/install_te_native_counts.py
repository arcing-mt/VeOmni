"""Process-only integer count workaround for a qualified MUSA shape family."""
import ast
from collections import Counter
import hashlib
import inspect
import json
import os
from pathlib import Path

import torch
import te_compact_prototype as reference

SOURCE_SHA = "7fdd07c124982745dd3ce311575a65108f2cf32e0b2d2c252ec8e7f0402417eb"
installed = False
COUNTERS = Counter()


def column_counts(mask, hidden, probabilities, experts):
    supported = (mask.device.type == "musa" and mask.device.index == torch.musa.current_device()
                 and mask.dtype == torch.bool and mask.ndim == 2
                 and mask.shape[1] == experts == 32
                 and (24577 <= mask.shape[0] <= 28608 or 28673 <= mask.shape[0] <= 32704)
                 and mask.is_contiguous() and mask.storage_offset() == 0
                 and hidden.dtype == torch.bfloat16 and hidden.ndim == 2
                 and hidden.shape == (mask.shape[0], 2048) and hidden.is_contiguous()
                 and hidden.device == mask.device and probabilities.device == mask.device
                 and probabilities.dtype == torch.float32
                 and probabilities.shape == (mask.shape[0], 8) and probabilities.is_contiguous())
    if supported:
        COUNTERS["Long_input_column_sum"] += 1
        # Preserve the integer contract. All duplicate/expert checks stay downstream.
        return mask.to(torch.int64).sum(0, dtype=torch.long)
    COUNTERS["original_Bool_column_sum"] += 1
    return mask.sum(0, dtype=torch.long)


def install():
    global installed
    if os.environ.get("TE_NATIVE_LONG_COUNTS", "0") != "1":
        return False
    if installed:
        return True
    if os.environ.get("TE_COMPACT", "0") != "1":
        raise RuntimeError("Integer count trial requires original TE compaction")
    import torch_musa
    if torch.__version__ != "2.11.0.post2" or torch_musa.__version__ != "2.11.0.post2+395c00c":
        raise RuntimeError("Integer sweep was qualified only on the frozen Torch/MUSA build")
    if hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest() != SOURCE_SHA:
        raise RuntimeError("Original TE compaction source changed")
    if reference._te_compact_permute.__code__.co_filename != reference.__file__:
        raise RuntimeError("Install integer count trial before any other reference transform")
    # Qualification files are frozen in the launcher manifest; validate their meaning too.
    sweep_path = Path(os.environ["TE_COUNTS_SUM_REPORT"])
    sweep = json.loads(sweep_path.read_text())
    if (sweep["status"] != "passed_integer_shape_sweep"
            or sweep["completed_pattern_shapes"] != 32808
            or sweep["admitted_integer_N_interval_inclusive"] != [24577, 32767]
            or len(sweep["actual_cases"]) != 10
            or not all(c["I64_input_equal"] and c["original_matches_captured"] for c in sweep["actual_cases"])):
        raise RuntimeError("Integer shape qualification is incomplete")
    source = inspect.getsource(reference._te_compact_permute)
    tree = ast.parse(source)
    fn = tree.body[0]
    target = ast.dump(ast.parse("counts = multi_hot.sum(0, dtype=torch.long)").body[0])
    matches = [i for i, node in enumerate(fn.body) if ast.dump(node) == target]
    if len(matches) != 1 or "_trial_native_column_counts" in reference.__dict__:
        raise RuntimeError("Original integer assignment did not match exactly")
    i = matches[0]
    fn.body[i] = ast.parse("counts = _trial_native_column_counts(multi_hot, recv_hidden, recv_probs, num_local_experts)").body[0]
    ast.fix_missing_locations(tree)
    reference.__dict__["_trial_native_column_counts"] = column_counts
    exec(compile(tree, "<reviewed_te_native_long_counts>", "exec"), reference.__dict__)
    installed = True
    return True


def stats():
    return {"installed": installed, "counters": dict(COUNTERS), "default_enabled": False,
            "scope": "Integer counts only, N24577..28608 or 28673..32704/E32/BF16H2048/current Torch/MUSA. "
                     "Correcting false rejection changes some Torch BF16 atomic fallback calls "
                     "to existing TE fixed FP32 accumulation; no old-fallback bitwise claim."}
