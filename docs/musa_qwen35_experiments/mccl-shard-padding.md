# [perf, dist] feat: archive guarded MCCL shard alignment

Pad eligible large contiguous BF16/FP32 shards to a 16-byte per-rank stride, use the original collective/dtype/op, then restore the logical layout. Copy/pack/wait/copy-back are measured. Scope: current Default FSDP2 comm, MUSA, 8 ranks, qualified streams. 48 operator cases/rank and controlled FSDP/AdamW updates passed. The TE+CE stage was 3.323586s and padding stage 3.071911s; this is directional full-step evidence, not a universal MCCL speed claim.

## Experimental archive contract

These are byte-for-byte publication copies of the recorded experiment, outside the installed veomni package. Normal tasks/entry points do not import them; all activation is explicit. No production API or default is added. Individual install() helpers must only be called deliberately in the recorded single-model MUSA process. Unsupported inputs retain their original guard/fallback.

Do not enable this archive in DPO, composed/multiple models, inference, another wheel, or another shape/layout merely because imports succeed. The qualification is limited to the documented target; portable runtime integration requires fresh gates.

## Files

- `experiments/musa/qwen35_20261010/install_mccl_padding.py`
- `experiments/musa/qwen35_20261010/check_mccl_padding.py`
- `experiments/musa/qwen35_20261010/check_fsdp_padding_updates.py`

## Dependencies and reproduction

Install the validated Torch2.11 post2/MUSA5.2 image and required native dependencies. Python helpers import their sibling modules from the flat archive directory, so add that directory to PYTHONPATH only for the explicit experiment. Original gate scripts may require recorded route/metadata fixtures outside Git; this publication does not include tensors, model weights or datasets.

For a component-only manual gate, use its listed check script after installing its declared sibling dependencies and providing the recorded external fixtures. A standalone component PR is not the combined training runner.

For the complete recorded counts+slot training experiment, compose all ten sibling branches: `gl/qwen35-payload-ready`, `gl/qwen35-lazy-clip-logging`, `gl/qwen35-supervised-ce`, `gl/qwen35-mccl-shard-padding`, `gl/qwen35-te-compaction`, `gl/qwen35-vision-csr`, `gl/qwen35-qk-preentry`, `gl/qwen35-native-warmup`, `gl/qwen35-te-long-counts`, `gl/qwen35-te-probability-slot`. The `gl/qwen35-te-probability-slot` PR supplies `run_qk_te_counts_slot_train.sh`, the shared trainer, and `recorded_qualifications/`; these are not included in every component PR. The frozen final launcher is tied to `/data/share/liang.geng/fsdp_overlap_test/VeOmni` and the original artifact/qualification paths. Only after that full composition exists, materialize the flat source files in `/data/share/liang.geng/fsdp_overlap_test/experiments/qwen35_optimize_20261009` and overlay its `recorded_qualifications/` tree there, refusing existing files with a different SHA. Preserve hashes, then explicitly run `bash run_qk_te_counts_slot_train.sh` only after all native dependencies and eight idle GPUs are verified. That script is an opt-in launcher and enables the experiments for that invocation.

Do not rewrite report hashes to make a different runtime look qualified. Porting directory/layout/installer logic requires separate verification; no GPU rerun was performed during this source publication.

## Validation

Publication checks compare source SHA, Python AST / shell syntax and repo make quality. Device evidence is historical, for the exact archived bytes. These MUSA-specific manual gates are not wired into the CUDA/NPU CI runner, and are not claimed as CI coverage. Source-receipt manifests and selected small JSON reports are included; raw training logs/caches/tensors are excluded.

## Related material

[PR3](https://github.com/arcing-mt/VeOmni/pull/3), [PR4](https://github.com/arcing-mt/VeOmni/pull/4), and [TE compatibility MR156](https://sh-code.mthreads.com/ai/TransformerEngine/-/merge_requests/156).
