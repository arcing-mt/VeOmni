"""Replace only TE probability construction/adjoint, with a source guard."""
import ast
import hashlib
import inspect
import os
from pathlib import Path

import te_compact_prototype as reference
import te_slot_probability

SOURCE_SHA = "7fdd07c124982745dd3ce311575a65108f2cf32e0b2d2c252ec8e7f0402417eb"
installed = False


def install():
    global installed
    if os.environ.get("TE_SLOT_PROBABILITY", "0") != "1":
        return False
    if installed:
        return True
    if os.environ.get("TE_COMPACT", "0") != "1":
        raise RuntimeError("Slot probability experiment requires original TE compaction")
    if hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest() != SOURCE_SHA:
        raise RuntimeError("Original TE compaction source does not match reviewed version")
    tree = ast.parse(inspect.getsource(reference._te_compact_permute))
    function = tree.body[0]
    expected = [
        "weights = torch.zeros((num_tokens, num_local_experts + 1), dtype=torch.float32, device=recv_hidden.device)",
        "weights.scatter_(1, (recv_indices + 1).to(torch.int64), recv_probs)",
        "weights = weights[:, 1:].contiguous()",
        "permuted, permuted_probs = _TECompactPermute.apply(recv_hidden, weights, row_nt, row_id_map, num_tokens, num_local_experts, num_out)",
    ]
    # Match syntax trees: Python 3.10 prints tuple targets with parentheses,
    # while newer Python versions omit them. Keep the structural guard exact.
    expected_nodes = [ast.dump(ast.parse(statement).body[0]) for statement in expected]
    starts = [i for i in range(len(function.body) - 3)
              if [ast.dump(node) for node in function.body[i:i + 4]] == expected_nodes]
    if len(starts) != 1:
        raise RuntimeError("Original TE probability statements do not match exactly")
    replacement = ast.parse("permuted, permuted_probs = _trial_slot_compact(recv_hidden, recv_probs, recv_indices, row_nt, row_id_map, num_tokens, num_local_experts, num_out)").body
    i = starts[0]
    function.body[i:i + 4] = replacement
    ast.fix_missing_locations(tree)
    if "_trial_slot_compact" in reference.__dict__:
        raise RuntimeError("Private slot helper already exists")
    # All original routing/counts/duplicate decisions and returns are intact.
    reference.__dict__["_trial_slot_compact"] = te_slot_probability.compact
    exec(compile(tree, "<reviewed_te_slot_probability>", "exec"), reference.__dict__)
    installed = True
    return True


def stats():
    return {"installed": installed, "counters": te_slot_probability.stats(),
            "scope": "Probability copy-only fusion; original counts/duplicate fallback and all TE hidden/merge native arithmetic unchanged. Original BF16 probability weighting unchanged. No new GPU failure fallback."}
