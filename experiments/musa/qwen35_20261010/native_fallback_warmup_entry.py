"""Opt-in initialization inside measured step1; native training is unchanged."""
import functools
import json
import os
from pathlib import Path
import runpy
import sys

import torch
from veomni.trainer.vlm_trainer import VLMTrainer
from native_fallback_warmup import warm_native_fallback

original = VLMTrainer.train_step
report = {"status": "disabled", "measured_step": None}
enabled = os.environ.get("NATIVE_FALLBACK_WARMUP", "0") == "1"


@functools.wraps(original)
def initialized(self, *args, **kwargs):
    global report
    if report["status"] == "disabled":
        if self.base.state.global_step != 0:
            raise RuntimeError("Native initialization experiment requires original step1")
        current_device = torch.musa.current_device()
        if current_device != int(os.environ["LOCAL_RANK"]):
            raise RuntimeError("Trainer MUSA device does not match local rank")
        report = {"status": "started", "measured_step": 1, "device": current_device}
        report.update(warm_native_fallback(torch.device("musa", current_device)))
    return original(self, *args, **kwargs)


if enabled:
    # The existing instrumented entry captures this wrapper, so it measures
    # initialization inside its original step1 timer.
    VLMTrainer.train_step = initialized
try:
    runpy.run_path(str(Path(__file__).with_name("instrumented_train_vlm.py")), run_name="__main__")
finally:
    training_had_error = sys.exc_info()[0] is not None
    VLMTrainer.train_step = original
    try:
        target = Path(os.environ["STEP_TIMING_DIR"]) / f"native_warmup_rank{os.environ.get('RANK', 'unknown')}.json"
        target.write_text(json.dumps(report, indent=2) + "\n")
    except Exception:
        if not training_had_error:
            raise
