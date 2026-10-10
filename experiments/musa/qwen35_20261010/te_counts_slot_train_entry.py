"""Opt-in counts/slot-copy composition; unchanged native warmup and measured trainer entry."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

import torch
import install_te_counts_slot as trial

ROOT = Path(__file__).resolve().parent
enabled = os.environ.get("TE_COUNTS_SLOT_COMBINED", "0") == "1"
qualification = {}
if enabled:
    qualification_files = json.loads((ROOT / "te_counts_slot_train_qualification_sha256.json").read_text())
    for label, item in qualification_files.items():
        assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"], label
    operator_path = Path(os.environ["TE_COUNTS_SLOT_GATE_REPORT"])
    operator = json.loads(operator_path.read_text())
    assert operator["status"] == "passed_operator_gates"
    expected_gate_sha = hashlib.sha256(operator_path.read_bytes()).hexdigest()
    for name, digest in operator["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    micro_path = Path(os.environ["TE_COUNTS_SLOT_ACTUAL_MICRO"])
    micro = json.loads(micro_path.read_text())
    assert micro["status"] == "completed_actual_routing_micro" and len(micro["cases"]) == 10
    assert micro["gate_sha256"] == expected_gate_sha
    assert all(row["measurements"][1]["median_ms"] < min(row["measurements"][0]["median_ms"], row["measurements"][2]["median_ms"])
               for row in micro["cases"])
    for name, digest in micro["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    fsdp_dir = Path(os.environ["TE_COUNTS_SLOT_FSDP_REPORT_DIR"])
    assert (fsdp_dir / "exit_code.txt").read_text().strip() == "0"
    fsdp_hashes = {}
    for rank in range(8):
        gate_path = fsdp_dir / f"te_counts_slot_fsdp_updates_rank{rank}.json"
        gate = json.loads(gate_path.read_text())
        assert gate["rank"] == rank
        fsdp_hashes[str(rank)] = hashlib.sha256(gate_path.read_bytes()).hexdigest()
        assert gate["status"] == "passed" and len(gate["results"]) == 6
        assert gate["operator_gate_sha256"] == expected_gate_sha
        assert gate["actual_micro_sha256"] == hashlib.sha256(micro_path.read_bytes()).hexdigest()
        assert not gate["deterministic_algorithms_enabled"]
        for name, digest in gate["source_sha256"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    for name, digest in json.loads((fsdp_dir / "runtime_source_sha256.json").read_text()).items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    assert os.environ.get("TE_SLOT_PROBABILITY", "0") == "1"
    assert os.environ.get("TE_NATIVE_LONG_COUNTS", "0") == "1"
    assert os.environ.get("TE_MERGE_CHECKS", "0") == "0"
    assert os.environ.get("TE_NATIVE_COUNTS_OBSERVE", "0") == "0"
    assert trial.install()
    qualification = {"qualification_file_sha256": {label: item["sha256"] for label, item in qualification_files.items()},
                     "operator_gate_sha256": expected_gate_sha,
                     "actual_micro_sha256": hashlib.sha256(micro_path.read_bytes()).hexdigest(),
                     "fsdp_report_sha256": fsdp_hashes, "source_sha256": operator["source_sha256"]}
try:
    runpy.run_path(str(ROOT / "native_fallback_warmup_entry.py"), run_name="__main__")
finally:
    training_had_error = sys.exc_info()[0] is not None
    try:
        destination = Path(os.environ["STEP_TIMING_DIR"]) / f"te_counts_slot_trial_rank{os.environ.get('RANK', 'unknown')}.json"
        destination.write_text(json.dumps({"enabled": enabled, "stats": trial.stats(),
                                          "rank": int(os.environ.get("RANK", "-1")),
                                          "entry_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                          "runtime_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                          **qualification}, indent=2) + "\n")
    except Exception:
        if not training_had_error:
            raise
