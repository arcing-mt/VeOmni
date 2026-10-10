"""Opt-in composition of independently qualified counts and probability copies."""
import ast
import hashlib
import json
import os
from pathlib import Path

import install_te_native_counts as counts
import te_compact_prototype as reference
import te_slot_probability as slot

installed = False
COUNTERS = counts.COUNTERS
SOURCE_SHA = "7fdd07c124982745dd3ce311575a65108f2cf32e0b2d2c252ec8e7f0402417eb"
COUNTS_SHA = "88a7ebc88d1d5a4927c37607a64588049bda796d9638c24ac302470be600d2af"
SLOT_SHA = "78fe144b7894ae2c9d78efab1eec67e4616b6a9b76a8e01bb20762358334c1c1"


def install():
    global installed
    if os.environ.get("TE_COUNTS_SLOT_COMBINED", "0") != "1":
        return False
    if installed:
        return True
    if os.environ.get("TE_NATIVE_LONG_COUNTS", "0") != "1" or os.environ.get("TE_SLOT_PROBABILITY", "0") != "1":
        raise RuntimeError("Composition requires explicit counts and slot opt-in")
    for module, expected in ((reference, SOURCE_SHA), (counts, COUNTS_SHA), (slot, SLOT_SHA)):
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Qualified component source changed")
    slot_gate = json.loads(Path(os.environ["TE_SLOT_GATE_REPORT"]).read_text())
    if slot_gate["status"] != "passed_operator_gates" or slot_gate["source_sha256"]["te_slot_probability.py"] != SLOT_SHA:
        raise RuntimeError("Probability component qualification missing")
    assert counts.install()
    if reference._te_compact_permute.__code__.co_filename != "<reviewed_te_native_long_counts>":
        raise RuntimeError("Unexpected count transform before composition")
    original = ast.parse(Path(reference.__file__).read_text())
    fn = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "_te_compact_permute")
    tree = ast.Module(body=[fn], type_ignores=[])
    target = ast.dump(ast.parse("counts = multi_hot.sum(0, dtype=torch.long)").body[0])
    found = [i for i, node in enumerate(fn.body) if ast.dump(node) == target]
    assert len(found) == 1
    fn.body[found[0]] = ast.parse("counts = _trial_native_column_counts(multi_hot, recv_hidden, recv_probs, num_local_experts)").body[0]
    expected = [
        "weights = torch.zeros((num_tokens, num_local_experts + 1), dtype=torch.float32, device=recv_hidden.device)",
        "weights.scatter_(1, (recv_indices + 1).to(torch.int64), recv_probs)",
        "weights = weights[:, 1:].contiguous()",
        "permuted, permuted_probs = _TECompactPermute.apply(recv_hidden, weights, row_nt, row_id_map, num_tokens, num_local_experts, num_out)",
    ]
    nodes = [ast.dump(ast.parse(s).body[0]) for s in expected]
    starts = [i for i in range(len(fn.body) - 3) if [ast.dump(n) for n in fn.body[i:i+4]] == nodes]
    if len(starts) != 1 or "_trial_slot_compact" in reference.__dict__:
        raise RuntimeError("Original probability statements did not match exactly")
    i = starts[0]
    fn.body[i:i+4] = ast.parse("permuted, permuted_probs = _trial_slot_compact(recv_hidden, recv_probs, recv_indices, row_nt, row_id_map, num_tokens, num_local_experts, num_out)").body
    reference.__dict__["_trial_slot_compact"] = slot.compact
    ast.fix_missing_locations(tree)
    exec(compile(tree, "<reviewed_te_counts_slot_composition>", "exec"), reference.__dict__)
    installed = True
    return True


def stats():
    return {"installed": installed, "counts": counts.stats(), "slot": slot.stats(),
            "scope": "Experimental composition, default off. Existing counts and slot guards remain; "
                     "corrected count decisions can change old fallback to intended TE math. "
                     "No combined operator/FSDP/full-training qualification implied."}
