---
orphan: true
---

# Opt-in MUSA FSDP2 collective alignment

Set `VEOMNI_MUSA_FSDP_SHARD_PADDING=1` before ordinary model parallelization. Each FSDP module in that model receives custom all-gather/reduce-scatter objects via PyTorch's `set_custom_*` APIs. Default `0` changes no collective objects. There is no global PyTorch method replacement. Only `TORCH_MUSA_FSDP2_OVERLAP_LEVEL=0` and `TORCH_MUSA_FSDP2_COMM_TYPE=0` are supported; other explicit selections fail before forward because vendor initialization can replace custom collectives.

Only contiguous 1D BF16/FP32 MUSA tensors on eight-rank groups, per-rank length≥1,000,000 and misaligned to 16 bytes, take the padded path. Padding extends every rank's block to a 16-byte boundary, preserving dtype and reduction op. Copy/wait/unpadding costs remain part of the collective call. Unsupported shapes keep the default path.

The dedicated `run_qwen35_35b_a3b_8card_train.sh` launcher enables the eight qualified performance flags across PRs #15/#16/#17 and `VEOMNI_MCCL_NATIVE_AVG` by default, matching the latest combined validation. Apply all three PRs to obtain the complete configuration. Set any individual flag to `0` to opt out; the eight new flags accept `0` or `1`. The launcher prints the selected flags and rejects nonzero FSDP overlap/comm modes while shard padding is enabled. Native fallback warmup also requires counting sort, enabled by this launcher; disabling counting sort with ACE requires `VEOMNI_MUSA_ACE_NATIVE_WARMUP=0`. General library defaults remain opt-in.

```bash
bash run_qwen35_35b_a3b_8card_train.sh
# Example: disable supervised CE for a comparison.
VEOMNI_MUSA_SUPERVISED_CE=0 bash run_qwen35_35b_a3b_8card_train.sh
```

Debug tensor formatting in ExtraParallel gradient clipping is deferred until a log is emitted. This avoids device scalar materialization when DEBUG logging is disabled; reduction groups, values and clipping order are unchanged.

An earlier directional stage comparison was ~3.3236 → ~3.0719 s/step with collective padding. Historical full composition across all three PR groups averaged 2.99248 / 2.99216 s over steps10–50; neither lazy logging nor this PR alone establishes <3s.

Tests cover rank-block packing/unpacking, BF16/FP32, sync/async waits, unchanged reduction operation, disabled/CPU fallback. Real eight-rank MUSA validation is additionally recorded in the PR body.

Reorganized runtime validation on the qualified image: eight ranks each completed 50 normal-entry steps, exit 0, with the selected core paths active. Fixed steps10–50 averaged 3.014633 s/step across rank means; the full50-step average including startup was 3.635762 s/step, logged peak 71.85GB. This run did not reproduce the archived <3s result. No isolated contribution or long-run/resume claim is made.

The unchanged repeat averaged 3.012666 s/step over the same steps10–50 window (full50 including startup: 3.637934 s/step). Both runs explicitly enabled the flags now defaulted by the dedicated launcher; changing the launcher defaults does not establish a new speedup.
