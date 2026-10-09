#!/usr/bin/env bash

# Qwen3.5-35B-A3B single-node training on 8x MTT S5000.
#
# Direct execution uses the best configuration validated on the 30 environment:
# DeepEP ACE + shared-expert overlap, FSDP2 overlap off, 32 MCCL channels,
# Vision Linear patch embed, foreach grad norm, and the tuned MUSA FLA backend.
#
#   bash run_qwen35_35b_a3b_8card_train.sh
#
# Common overrides:
#   MODEL_PATH=/path/to/model DATA_PATH=/path/to/data.json \
#   STEPS_PER_EPOCH=20 NUM_TRAIN_EPOCHS=1 \
#   bash run_qwen35_35b_a3b_8card_train.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

normalize_bool() {
  local name="$1"
  local value="${!name}"
  case "${value,,}" in
    0|false|no|off) printf -v "${name}" '%s' false ;;
    1|true|yes|on) printf -v "${name}" '%s' true ;;
    *) die "${name} must be a boolean value: ${value}" ;;
  esac
}

require_choice() {
  local name="$1"
  local value="${!name}"
  shift
  local candidate
  for candidate in "$@"; do
    [[ "${value}" == "${candidate}" ]] && return
  done
  die "${name} must be one of: $*; got '${value}'."
}

require_positive_integer() {
  local name="$1"
  local value="${!name}"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer: ${value}"
}

set_defaults() {
  # Runtime and devices.
  export OMP_NUM_THREADS=4
  export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  export MUSA_KERNEL_TIMEOUT=3200000
  export ACCELERATOR_BACKEND="musa"
  export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
  export MATE_MUSA_ARCH_LIST="${MATE_MUSA_ARCH_LIST:-3.1}"
  PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-/usr/bin/python}}"
  export TORCHRUN_PYTHON="${TORCHRUN_PYTHON:-${PYTHON_BIN}}"
  MAX_EXISTING_MIB="${MAX_EXISTING_MIB:-0}"

  if [[ -d /usr/local/mtshmem/lib ]]; then
    export LD_LIBRARY_PATH="/usr/local/mtshmem/lib:${LD_LIBRARY_PATH:-}"
  fi

  # Validated MCCL defaults. Native AVG remains available for explicit A/B,
  # but two 50-step runs showed no repeatable gain over SUM + scale.
  export MCCL_CTA_POLICY="${MCCL_CTA_POLICY:-2}"
  export MCCL_PROTOS="${MCCL_PROTOS:-2}"
  export MCCL_ALGOS="${MCCL_ALGOS:-1}"
  export MCCL_MAX_NCHANNELS="${MCCL_MAX_NCHANNELS:-32}"
  export MCCL_MIN_NCHANNELS="${MCCL_MIN_NCHANNELS:-32}"
  export VEOMNI_MCCL_NATIVE_AVG="${VEOMNI_MCCL_NATIVE_AVG:-0}"
  export VEOMNI_MUSA_FSDP2_FOREACH_GRAD_NORM="${VEOMNI_MUSA_FSDP2_FOREACH_GRAD_NORM:-1}"

  # MoE and FSDP. The direct-run production path is ACE with FSDP overlap off.
  MOE_DISPATCHER="${MOE_DISPATCHER:-deepep_ace}"
  MOE_DEEPEP_NUM_SMS="${MOE_DEEPEP_NUM_SMS:-20}"
  MOE_DEEPEP_TOKEN_CAPACITY="${MOE_DEEPEP_TOKEN_CAPACITY:-8192}"
  MOE_SHARED_EXPERT_OVERLAP="${MOE_SHARED_EXPERT_OVERLAP:-true}"
  VEOMNI_MUSA_DEEPEP_COUNTING_SORT="${VEOMNI_MUSA_DEEPEP_COUNTING_SORT:-true}"
  FSDP_FORWARD_PREFETCH="${FSDP_FORWARD_PREFETCH:-true}"
  FSDP_BACKWARD_PREFETCH="${FSDP_BACKWARD_PREFETCH:-true}"
  FSDP_DEEPEP_STREAM_COMPAT="${FSDP_DEEPEP_STREAM_COMPAT:-false}"
  FSDP_DEEPEP_SHARED_COMM_STREAM="${FSDP_DEEPEP_SHARED_COMM_STREAM:-false}"
  TORCH_MUSA_FSDP2_COMM_TYPE="${TORCH_MUSA_FSDP2_COMM_TYPE:-0}"
  TORCH_MUSA_FSDP2_OVERLAP_LEVEL="${TORCH_MUSA_FSDP2_OVERLAP_LEVEL:-0}"

  # Model kernels and input pipeline.
  ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_3}"
  RMS_NORM_GATED_IMPLEMENTATION="${RMS_NORM_GATED_IMPLEMENTATION:-fla}"
  CAUSAL_CONV1D_IMPLEMENTATION="${CAUSAL_CONV1D_IMPLEMENTATION:-fla}"
  CHUNK_GATED_DELTA_RULE_IMPLEMENTATION="${CHUNK_GATED_DELTA_RULE_IMPLEMENTATION:-}"
  VISION_PATCH_EMBED_IMPLEMENTATION="${VISION_PATCH_EMBED_IMPLEMENTATION:-linear}"
  SKIP_EMPTY_MODALITY_DUMMY="${SKIP_EMPTY_MODALITY_DUMMY:-true}"
  DATALOADER_USE_BACKGROUND_PREFETCHER="${DATALOADER_USE_BACKGROUND_PREFETCHER:-false}"
  SYNC_EACH_TRAIN_STEP="${SYNC_EACH_TRAIN_STEP:-true}"

  # Model, data, and run length.
  export MODEL_PATH="${MODEL_PATH:-/data/share/models/Qwen3.5-35B-A3B}"
  export DATA_PATH="${DATA_PATH:-/data/share/liang.geng/fsdp_overlap_test/data/sharegpt4v_coco_full/sharegpt4v_coco_full.json}"
  DATA_TYPE="${DATA_TYPE:-conversation}"
  TEXT_KEYS="${TEXT_KEYS:-messages}"
  MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
  TRAIN_SIZE="${TRAIN_SIZE:-52428800}"
  MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
  GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((MICRO_BATCH_SIZE * NPROC_PER_NODE))}"
  STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-50}"
  NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"

  # Optional profiler window.
  PROFILE_ENABLE="${PROFILE_ENABLE:-false}"
  PROFILE_START_STEP="${PROFILE_START_STEP:-3}"
  PROFILE_END_STEP="${PROFILE_END_STEP:-6}"
  PROFILE_TRACE_DIR="${PROFILE_TRACE_DIR:-${SCRIPT_DIR}/traces/qwen35a3b_image_$(date +%Y%m%d_%H%M%S)}"
  PROFILE_RECORD_SHAPES="${PROFILE_RECORD_SHAPES:-false}"
  PROFILE_MEMORY="${PROFILE_MEMORY:-false}"
  PROFILE_STACK="${PROFILE_STACK:-false}"
  PROFILE_MODULES="${PROFILE_MODULES:-false}"
  PROFILE_RANK0_ONLY="${PROFILE_RANK0_ONLY:-true}"
}

resolve_gdn_backend() {
  [[ -n "${CHUNK_GATED_DELTA_RULE_IMPLEMENTATION}" ]] && return
  command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Python is not executable: ${PYTHON_BIN}"
  if "${PYTHON_BIN}" -c 'from torch_kernels.attention import gated_delta_net' >/dev/null 2>&1; then
    CHUNK_GATED_DELTA_RULE_IMPLEMENTATION=musa_tilelang
  else
    CHUNK_GATED_DELTA_RULE_IMPLEMENTATION=musa
  fi
}

validate_config() {
  command -v mthreads-gmi >/dev/null 2>&1 || die "mthreads-gmi is required for the card preflight."
  command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Python is not executable: ${PYTHON_BIN}"

  local name
  for name in MOE_SHARED_EXPERT_OVERLAP FSDP_FORWARD_PREFETCH FSDP_BACKWARD_PREFETCH \
    FSDP_DEEPEP_STREAM_COMPAT FSDP_DEEPEP_SHARED_COMM_STREAM \
    DATALOADER_USE_BACKGROUND_PREFETCHER SYNC_EACH_TRAIN_STEP \
    SKIP_EMPTY_MODALITY_DUMMY VEOMNI_MUSA_DEEPEP_COUNTING_SORT PROFILE_ENABLE; do
    normalize_bool "${name}"
  done

  require_choice MOE_DISPATCHER alltoall deepep deepep_ace
  require_choice VISION_PATCH_EMBED_IMPLEMENTATION conv3d linear
  require_choice CHUNK_GATED_DELTA_RULE_IMPLEMENTATION fla musa musa_tilelang
  for name in MOE_DEEPEP_NUM_SMS MOE_DEEPEP_TOKEN_CAPACITY MICRO_BATCH_SIZE \
    GLOBAL_BATCH_SIZE STEPS_PER_EPOCH NUM_TRAIN_EPOCHS PROFILE_START_STEP PROFILE_END_STEP; do
    require_positive_integer "${name}"
  done

  (( MOE_DEEPEP_NUM_SMS % 2 == 0 )) || die "MOE_DEEPEP_NUM_SMS must be even: ${MOE_DEEPEP_NUM_SMS}"
  [[ "${NPROC_PER_NODE}" == 8 ]] || die "this launcher requires NPROC_PER_NODE=8; got '${NPROC_PER_NODE}'."
  (( PROFILE_END_STEP > PROFILE_START_STEP )) || die "PROFILE_END_STEP must be greater than PROFILE_START_STEP."
  (( GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * NPROC_PER_NODE) == 0 )) || \
    die "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be a multiple of MICRO_BATCH_SIZE*NPROC_PER_NODE=$((MICRO_BATCH_SIZE * NPROC_PER_NODE))."

  IFS=',' read -r -a MUSA_CARDS <<< "${MUSA_VISIBLE_DEVICES}"
  [[ "${#MUSA_CARDS[@]}" -eq 8 ]] || \
    die "MUSA_VISIBLE_DEVICES must contain exactly eight cards; got '${MUSA_VISIBLE_DEVICES}'."
  local card
  for card in "${MUSA_CARDS[@]}"; do
    [[ "${card}" =~ ^[0-9]+$ ]] || die "invalid MUSA card id '${card}'."
  done

  [[ -d "${MODEL_PATH}" ]] || die "model path does not exist: ${MODEL_PATH}"
  [[ -f "${DATA_PATH}" ]] || die "prepared multimodal annotation does not exist: ${DATA_PATH}"

  local total_train_steps=$((10#${STEPS_PER_EPOCH} * 10#${NUM_TRAIN_EPOCHS}))
  if [[ "${PROFILE_ENABLE}" == true ]] && (( total_train_steps < 10#${PROFILE_END_STEP} )); then
    die "profiling ends at step ${PROFILE_END_STEP}, but this run has at most ${total_train_steps} steps."
  fi

  validate_deepep_fsdp_combination
}

validate_deepep_fsdp_combination() {
  if [[ "${MOE_SHARED_EXPERT_OVERLAP}" == true && "${MOE_DISPATCHER}" == alltoall ]]; then
    die "MOE_SHARED_EXPERT_OVERLAP requires a DeepEP dispatcher."
  fi

  if [[ "${FSDP_DEEPEP_STREAM_COMPAT}" == true ]]; then
    [[ "${MOE_DISPATCHER}" != alltoall ]] || die "FSDP_DEEPEP_STREAM_COMPAT requires a DeepEP dispatcher."
    [[ "${TORCH_MUSA_FSDP2_COMM_TYPE}" == 0 && "${TORCH_MUSA_FSDP2_OVERLAP_LEVEL}" == 2 ]] || \
      die "FSDP_DEEPEP_STREAM_COMPAT is validated only with COMM_TYPE=0 and OVERLAP_LEVEL=2."
  elif [[ "${MOE_DISPATCHER}" == deepep_ace && "${TORCH_MUSA_FSDP2_OVERLAP_LEVEL}" == 2 ]]; then
    die "raw FSDP2 level 2 deadlocks with DeepEP-ACE; set FSDP_DEEPEP_STREAM_COMPAT=true."
  fi

  if [[ "${FSDP_DEEPEP_SHARED_COMM_STREAM}" == true && "${FSDP_DEEPEP_STREAM_COMPAT}" != true ]]; then
    die "FSDP_DEEPEP_SHARED_COMM_STREAM requires FSDP_DEEPEP_STREAM_COMPAT=true."
  fi

  if [[ "${MOE_DISPATCHER}" == deepep && "${TORCH_MUSA_FSDP2_OVERLAP_LEVEL}" != 0 && \
        "${FSDP_DEEPEP_SHARED_COMM_STREAM}" != true ]]; then
    die "standard DeepEP with FSDP2 overlap requires level 2 plus both DeepEP stream compatibility switches."
  fi
}

preflight_python() {
  "${PYTHON_BIN}" - <<'PY' || die "${PYTHON_BIN} is not a usable eight-card torch_musa environment."
import torch
import torch_musa  # noqa: F401

if not hasattr(torch, "musa") or not torch.musa.is_available():
    raise RuntimeError("torch_musa imported, but torch.musa is unavailable")
if torch.musa.device_count() < 8:
    raise RuntimeError(f"eight MUSA devices are required, but Python sees {torch.musa.device_count()}")
print(f"torch_musa preflight: torch={torch.__version__}, devices={torch.musa.device_count()}")
PY
}

preflight_deepep() {
  [[ "${MOE_DISPATCHER}" != alltoall ]] || return 0
  "${PYTHON_BIN}" - "${MOE_DISPATCHER}" <<'PY' || die "${MOE_DISPATCHER} requires a compatible DeepEP wheel."
import importlib.metadata as metadata
import inspect
import sys

from deep_ep import Buffer, EventOverlap  # noqa: F401
from deep_ep_cpp import EventHandle  # noqa: F401

if sys.argv[1] == "deepep_ace":
    required = {"use_ace", "token_num", "hidden_size", "num_topk"}
    missing = required.difference(inspect.signature(Buffer).parameters)
    if missing:
        raise RuntimeError(f"DeepEP Buffer is missing ACE parameters: {sorted(missing)}")
print(f"DeepEP preflight: deep_ep={metadata.version('deep_ep')}, dispatcher={sys.argv[1]}")
PY
}

preflight_cards() {
  local card used_mib
  for card in "${MUSA_CARDS[@]}"; do
    used_mib="$({
      mthreads-gmi --query --id "${card}" --display MEMORY,UTILIZATION --json
    } | "${PYTHON_BIN}" -c 'import json, re, sys; d=json.load(sys.stdin); s=d["GPU"][0]["FB Memory Usage"]["Used"]; print(int(re.search(r"[0-9]+", s).group()))')"
    if (( used_mib > MAX_EXISTING_MIB )); then
      echo "REFUSE: MUSA card ${card} uses ${used_mib} MiB (limit ${MAX_EXISTING_MIB} MiB)." >&2
      exit 3
    fi
    echo "MUSA card ${card}: ${used_mib} MiB in use; accepted by preflight."
  done
}

prepare_output() {
  CLEANUP_OUTPUT=0
  if [[ -z "${OUTPUT_DIR:-}" ]]; then
    OUTPUT_DIR="$(mktemp -d /tmp/veomni_qwen35_image_8card_smoke.XXXXXX)"
    CLEANUP_OUTPUT=1
  fi
}

cleanup_output() {
  if [[ "${CLEANUP_OUTPUT:-0}" -eq 1 && "${KEEP_OUTPUT:-0}" != 1 && -d "${OUTPUT_DIR:-}" ]]; then
    rm -rf -- "${OUTPUT_DIR}"
  fi
}

build_profile_args() {
  PROFILE_ARGS=(--train.profile.enable false)
  [[ "${PROFILE_ENABLE}" == true ]] || return 0
  mkdir -p "${PROFILE_TRACE_DIR}"
  PROFILE_ARGS=(
    --train.profile.enable true
    --train.profile.start_step "${PROFILE_START_STEP}"
    --train.profile.end_step "${PROFILE_END_STEP}"
    --train.profile.trace_dir "${PROFILE_TRACE_DIR}"
    --train.profile.record_shapes "${PROFILE_RECORD_SHAPES}"
    --train.profile.profile_memory "${PROFILE_MEMORY}"
    --train.profile.with_stack "${PROFILE_STACK}"
    --train.profile.with_modules "${PROFILE_MODULES}"
    --train.profile.rank0_only "${PROFILE_RANK0_ONLY}"
  )
}

print_run_config() {
  cat <<EOF
Starting Qwen3.5-35B-A3B eight-card image training.

Input
  model=${MODEL_PATH}
  data=${DATA_PATH}
  sequence=${MAX_SEQ_LEN}, micro_batch=${MICRO_BATCH_SIZE}, global_batch=${GLOBAL_BATCH_SIZE}
  steps_per_epoch=${STEPS_PER_EPOCH}, epochs=${NUM_TRAIN_EPOCHS}

Parallelism and communication
  devices=${MUSA_VISIBLE_DEVICES}; FSDP=8 (fsdp2), EP=8, SP=1
  dispatcher=${MOE_DISPATCHER}
  fsdp_overlap_level=${TORCH_MUSA_FSDP2_OVERLAP_LEVEL}, comm_type=${TORCH_MUSA_FSDP2_COMM_TYPE}
  fsdp_prefetch_forward=${FSDP_FORWARD_PREFETCH}, fsdp_prefetch_backward=${FSDP_BACKWARD_PREFETCH}
  deepep_stream_compat=${FSDP_DEEPEP_STREAM_COMPAT}, shared_comm_stream=${FSDP_DEEPEP_SHARED_COMM_STREAM}
  mccl_channels=${MCCL_MIN_NCHANNELS}-${MCCL_MAX_NCHANNELS}, protocols=${MCCL_PROTOS}, algorithms=${MCCL_ALGOS}, cta_policy=${MCCL_CTA_POLICY}
  mccl_buffer=${MCCL_BUFFSIZE:-auto}, block_schedule=${MUSA_BLOCK_SCHEDULE_MODE:-auto}, ib_gid=${MCCL_IB_GID_INDEX:-auto}, shared_buffers=${MCCL_NET_SHARED_BUFFERS:-auto}
EOF

  if [[ "${MOE_DISPATCHER}" != alltoall ]]; then
    echo "  DeepEP: sms=${MOE_DEEPEP_NUM_SMS}, shared_expert_overlap=${MOE_SHARED_EXPERT_OVERLAP}, counting_sort=${VEOMNI_MUSA_DEEPEP_COUNTING_SORT}"
  fi
  if [[ "${MOE_DISPATCHER}" == deepep_ace ]]; then
    echo "  DeepEP ACE: token_capacity=${MOE_DEEPEP_TOKEN_CAPACITY}"
  fi

  cat <<EOF

Kernels and pipeline
  attention=${ATTN_IMPLEMENTATION}, gdn=${CHUNK_GATED_DELTA_RULE_IMPLEMENTATION}
  gated_rms_norm=${RMS_NORM_GATED_IMPLEMENTATION}, causal_conv1d=${CAUSAL_CONV1D_IMPLEMENTATION}
  vision_patch_embed=${VISION_PATCH_EMBED_IMPLEMENTATION}, skip_empty_dummy=${SKIP_EMPTY_MODALITY_DUMMY}
  foreach_grad_norm=${VEOMNI_MUSA_FSDP2_FOREACH_GRAD_NORM}, native_mccl_avg=${VEOMNI_MCCL_NATIVE_AVG}
  background_prefetch=${DATALOADER_USE_BACKGROUND_PREFETCHER}, sync_each_step=${SYNC_EACH_TRAIN_STEP}

Output
  output_dir=${OUTPUT_DIR}
  profile=${PROFILE_ENABLE}
EOF

  if [[ "${PROFILE_ENABLE}" == true ]]; then
    echo "  trace_dir=${PROFILE_TRACE_DIR}"
  fi
}

run_training() {
  export TORCH_MUSA_FSDP2_COMM_TYPE
  export TORCH_MUSA_FSDP2_OVERLAP_LEVEL
  export VEOMNI_MUSA_DEEPEP_FSDP_STREAM_COMPAT="${FSDP_DEEPEP_STREAM_COMPAT}"
  export VEOMNI_MUSA_DEEPEP_FSDP_SHARED_COMM_STREAM="${FSDP_DEEPEP_SHARED_COMM_STREAM}"
  export VEOMNI_MUSA_DEEPEP_COUNTING_SORT

  local train_args=(
    tasks/train_vlm.py
    configs/multimodal/qwen3_5_moe/qwen3_5_moe_vl.yaml
    --model.model_path "${MODEL_PATH}"
    --data.train_path "${DATA_PATH}"
    --data.data_type "${DATA_TYPE}"
    --data.text_keys "${TEXT_KEYS}"
    --data.source_name sharegpt4v_sft
    --data.train_size "${TRAIN_SIZE}"
    --data.max_seq_len "${MAX_SEQ_LEN}"
    --data.dataloader.pin_memory false
    --data.dataloader.num_workers 8
    --data.dataloader.prefetch_factor 4
    --data.dataloader.persistent_workers true
    --data.dataloader.use_background_prefetcher "${DATALOADER_USE_BACKGROUND_PREFETCHER}"
    --train.max_steps "${STEPS_PER_EPOCH}"
    --train.num_train_epochs "${NUM_TRAIN_EPOCHS}"
    --train.global_batch_size "${GLOBAL_BATCH_SIZE}"
    --train.micro_batch_size "${MICRO_BATCH_SIZE}"
    --train.sync_each_train_step "${SYNC_EACH_TRAIN_STEP}"
    --model.ops_implementation.attn_implementation "${ATTN_IMPLEMENTATION}"
    --model.ops_implementation.rms_norm_implementation musa
    --model.ops_implementation.moe_implementation fused_musa
    --model.ops_implementation.moe_dispatcher "${MOE_DISPATCHER}"
    --model.ops_implementation.moe_deepep_num_sms "${MOE_DEEPEP_NUM_SMS}"
    --model.ops_implementation.moe_deepep_token_capacity "${MOE_DEEPEP_TOKEN_CAPACITY}"
    --model.ops_implementation.moe_shared_expert_overlap "${MOE_SHARED_EXPERT_OVERLAP}"
    --model.ops_implementation.skip_empty_modality_dummy "${SKIP_EMPTY_MODALITY_DUMMY}"
    --model.ops_implementation.vision_patch_embed_implementation "${VISION_PATCH_EMBED_IMPLEMENTATION}"
    --model.ops_implementation.rotary_pos_emb_implementation eager
    --model.ops_implementation.rotary_pos_emb_vision_implementation musa
    --model.ops_implementation.cross_entropy_loss_implementation chunk_loss
    --model.ops_implementation.rms_norm_gated_implementation "${RMS_NORM_GATED_IMPLEMENTATION}"
    --model.ops_implementation.causal_conv1d_implementation "${CAUSAL_CONV1D_IMPLEMENTATION}"
    --model.ops_implementation.chunk_gated_delta_rule_implementation "${CHUNK_GATED_DELTA_RULE_IMPLEMENTATION}"
    --model.accelerator.gradient_checkpointing.enable true
    --model.accelerator.dp_shard_size 8
    --model.accelerator.ep_size 8
    --model.accelerator.ulysses_size 1
    --model.accelerator.fsdp_config.fsdp_mode fsdp2
    --model.accelerator.fsdp_config.forward_prefetch "${FSDP_FORWARD_PREFETCH}"
    --model.accelerator.fsdp_config.backward_prefetch "${FSDP_BACKWARD_PREFETCH}"
    --model.accelerator.init_device meta
    --train.checkpoint.output_dir "${OUTPUT_DIR}"
    --train.checkpoint.save_steps 0
    --train.checkpoint.save_epochs 0
    --train.wandb.enable false
  )

  bash train.sh "${train_args[@]}" "${PROFILE_ARGS[@]}"
}

main() {
  set_defaults
  resolve_gdn_backend
  validate_config
  preflight_python
  preflight_deepep
  preflight_cards
  prepare_output
  trap cleanup_output EXIT
  build_profile_args
  print_run_config
  run_training
}

main "$@"
