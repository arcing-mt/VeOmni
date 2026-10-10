#!/usr/bin/env bash
set -euo pipefail
trial_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qk_te_counts_slot}"
export TE_COMPACT=1 SUPERVISED_CE=1 MCCL_SHARD_PADDING=1 VEOMNI_MCCL_NATIVE_AVG=1
export MOE_SHARED_EXPERT_OVERLAP=true SHARED_SCHEDULE_DIAGNOSTIC=1
export FLA_CONV_VJP=0 QK_PREENTRY=1
export FSDP_FUSED_AG_PACK=0 FSDP_ALIGNED_AG_PACK=0 DATALOADER_USE_BACKGROUND_PREFETCHER=false
export HOST_CU_CACHE=0 VISION_EMBEDDING_CSR=1 VISION_GEOMETRY_CACHE=0
export SUPERVISED_CE_CHUNK2048=0 DEFERRED_ADAM_CLIP=0 FSDP_PREFETCH_DEPTH=1
export DEFER_TRAIN_SCALARS=0 EARLY_RECURRENT_MASK=0
export TE_MERGE_CHECKS=0 TE_FUSED_UNPERMUTE=0 TEXT_RETAIN_EVERY5=0 VISION_RETAIN_THIRD=0
export FUSED_UNPERMUTE=0 SCALAR_COMPACT=0
export LIGHT_ROUTING_DIAGNOSTIC=0 PAIRED_ROUTING_DIAGNOSTIC=0 CAPTURE_OP_METADATA=0
export MCCL_CTA_POLICY=2 MCCL_MIN_NCHANNELS=32 MCCL_MAX_NCHANNELS=32 MCCL_PROTOS=2 MCCL_ALGOS=1
unset MCCL_BUFFSIZE MCCL_PROTO MCCL_ALGO MCCL_MIN_P2P_NCHANNELS MCCL_MAX_P2P_NCHANNELS
export PROFILE_ENABLE=false STEPS_PER_EPOCH=50
export NATIVE_FALLBACK_WARMUP=1
export TE_COUNTS_SLOT_COMBINED=1 TE_NATIVE_LONG_COUNTS=1 TE_SLOT_PROBABILITY=1 TE_NATIVE_COUNTS_OBSERVE=0
export TE_COUNTS_SUM_REPORT="${trial_dir}/te_native_sum_sweep_20261010_055717/native_count_sum_report.json"
export TE_COUNTS_SLOT_GATE_REPORT="${trial_dir}/te_counts_slot_gate_20261010_064351/te_counts_slot_gate.json"
export TE_COUNTS_SLOT_ACTUAL_MICRO="${trial_dir}/te_counts_slot_actual_micro_20261010_064655/te_counts_slot_actual_bench.json"
export TE_COUNTS_SLOT_FSDP_REPORT_DIR="${trial_dir}/te_counts_slot_fsdp_20261010_064911"
export TE_SLOT_GATE_REPORT="${trial_dir}/te_slot_probability_gate.json"
export TIMING_ENTRY_SCRIPT=te_counts_slot_train_entry.py
sha256sum "${trial_dir}/install_te_counts_slot.py" "${trial_dir}/install_te_native_counts.py" "${trial_dir}/te_slot_probability.py" "${trial_dir}/te_counts_slot_train_entry.py" "${trial_dir}/run_qk_te_counts_slot_train.sh"
sha256sum "${trial_dir}/native_fallback_warmup.py" "${trial_dir}/native_fallback_warmup_entry.py" \
  "${trial_dir}/install_qk_preentry.py" "${trial_dir}/qk_repeat_l2norm.py" \
  "${trial_dir}/qk_repeat_l2norm_kernel.py" "${trial_dir}/instrumented_train_vlm.py" "${trial_dir}/metadata_cache.py" \
  "${trial_dir}/vision_embedding_csr.py" "${trial_dir}/install_vision_embedding_csr.py" \
  "${trial_dir}/install_te_compact.py" "${trial_dir}/te_compact_prototype.py" \
  "${trial_dir}/install_mccl_padding.py" "${trial_dir}/supervised_chunk_loss.py" \
  veomni/distributed/fsdp2/clip_grad_norm.py veomni/ops/kernels/gated_delta_rule/musa_tilelang.py
exec bash "${trial_dir}/run_experiment.sh"
