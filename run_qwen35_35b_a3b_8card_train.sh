#!/usr/bin/env bash

# Safe single-node, eight-card Qwen3.5-35B-A3B image-training smoke test on MUSA.
#
# The launcher uses the Qwen3.5-MoE VLM config with the validated MUSA
# parallel, operator and batch settings.
#
# Usage:
#   MODEL_PATH=/path/to/Qwen3.5-35B-A3B \
#   DATA_PATH=/path/to/sharegpt4v_coco.json \
#   bash run_qwen35_35b_a3b_8card_train.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if ! command -v mthreads-gmi >/dev/null 2>&1; then
  echo "ERROR: mthreads-gmi is required for the card-availability preflight." >&2
  exit 2
fi

export OMP_NUM_THREADS=4
export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MUSA_KERNEL_TIMEOUT=3200000
export ACCELERATOR_BACKEND="musa"
if [[ -d /usr/local/mtshmem/lib ]]; then
  export LD_LIBRARY_PATH="/usr/local/mtshmem/lib:${LD_LIBRARY_PATH:-}"
fi
export MCCL_CTA_POLICY="${MCCL_CTA_POLICY:-2}"
export MCCL_PROTOS="${MCCL_PROTOS:-2}"
export MCCL_ALGOS="${MCCL_ALGOS:-1}"
# Two 50-step runs were indistinguishable by cumulative stable-step time;
# the native-AVG run's final smoothed rate was ~0.02 s/step slower. Keep the
# established SUM+scale path by default and retain native AVG for explicit A/B.
export VEOMNI_MCCL_NATIVE_AVG="${VEOMNI_MCCL_NATIVE_AVG:-0}"
export VEOMNI_MUSA_FSDP2_FOREACH_GRAD_NORM="${VEOMNI_MUSA_FSDP2_FOREACH_GRAD_NORM:-1}"
# export MCCL_BUFFSIZE=20971520
# export MUSA_BLOCK_SCHEDULE_MODE=1
# export MCCL_IB_GID_INDEX=3
# export MCCL_NET_SHARED_BUFFERS=0
# Two repeated 20-step runs on 8x S5000 selected 32 channels over
# auto/8/16/24/48/64 for this FSDP2 + DeepEP workload.
export MCCL_MAX_NCHANNELS="${MCCL_MAX_NCHANNELS:-32}"
export MCCL_MIN_NCHANNELS="${MCCL_MIN_NCHANNELS:-32}"
IFS=',' read -r -a _MUSA_CARDS <<< "${MUSA_VISIBLE_DEVICES}"
if [[ "${#_MUSA_CARDS[@]}" -ne 8 ]]; then
  echo "ERROR: MUSA_VISIBLE_DEVICES must contain exactly eight cards for this launcher; got '${MUSA_VISIBLE_DEVICES}'." >&2
  exit 2
fi

# Refuse every non-zero allocation by default. Override only with an explicit
# assignment from the card owner.
MAX_EXISTING_MIB="${MAX_EXISTING_MIB:-0}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-/usr/bin/python}}"
MOE_DISPATCHER="${MOE_DISPATCHER:-deepep_ace}"
# Keep the DeepEP package default. Callers can override it with an even value
# for explicit A/B tests.
MOE_DEEPEP_NUM_SMS="${MOE_DEEPEP_NUM_SMS:-20}"
MOE_DEEPEP_TOKEN_CAPACITY="${MOE_DEEPEP_TOKEN_CAPACITY:-8192}"
MOE_SHARED_EXPERT_OVERLAP="${MOE_SHARED_EXPERT_OVERLAP:-true}"
FSDP_FORWARD_PREFETCH="${FSDP_FORWARD_PREFETCH:-true}"
FSDP_BACKWARD_PREFETCH="${FSDP_BACKWARD_PREFETCH:-true}"
FSDP_DEEPEP_STREAM_COMPAT="${FSDP_DEEPEP_STREAM_COMPAT:-false}"
FSDP_DEEPEP_SHARED_COMM_STREAM="${FSDP_DEEPEP_SHARED_COMM_STREAM:-false}"
DATALOADER_USE_BACKGROUND_PREFETCHER="${DATALOADER_USE_BACKGROUND_PREFETCHER:-false}"
SYNC_EACH_TRAIN_STEP="${SYNC_EACH_TRAIN_STEP:-true}"
SKIP_EMPTY_MODALITY_DUMMY="${SKIP_EMPTY_MODALITY_DUMMY:-true}"
VISION_PATCH_EMBED_IMPLEMENTATION="${VISION_PATCH_EMBED_IMPLEMENTATION:-linear}"
CHUNK_GATED_DELTA_RULE_IMPLEMENTATION="${CHUNK_GATED_DELTA_RULE_IMPLEMENTATION:-}"
RMS_NORM_GATED_IMPLEMENTATION="${RMS_NORM_GATED_IMPLEMENTATION:-fla}"
CAUSAL_CONV1D_IMPLEMENTATION="${CAUSAL_CONV1D_IMPLEMENTATION:-fla}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python interpreter does not exist or is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ -z "${CHUNK_GATED_DELTA_RULE_IMPLEMENTATION}" ]]; then
  if "${PYTHON_BIN}" -c 'from torch_kernels.attention import gated_delta_net' >/dev/null 2>&1; then
    CHUNK_GATED_DELTA_RULE_IMPLEMENTATION=musa_tilelang
  else
    # The production image currently has tuned FLA but no torch_kernels package.
    CHUNK_GATED_DELTA_RULE_IMPLEMENTATION=musa
  fi
fi

if ! "${PYTHON_BIN}" - <<'PY'
import torch
import torch_musa  # noqa: F401  registers torch.musa

if not hasattr(torch, "musa") or not torch.musa.is_available():
    raise RuntimeError("torch_musa imported, but torch.musa is unavailable")

device_count = torch.musa.device_count()
if device_count < 8:
    raise RuntimeError(f"eight MUSA devices are required, but Python sees {device_count}")

print(f"torch_musa preflight: torch={torch.__version__}, devices={device_count}")
PY
then
  echo "ERROR: ${PYTHON_BIN} is not a usable eight-card torch_musa environment." >&2
  exit 2
fi

if [[ "${MOE_DISPATCHER}" != "alltoall" && "${MOE_DISPATCHER}" != "deepep_ace" ]]; then
  echo "ERROR: MOE_DISPATCHER must be alltoall or deepep_ace; got '${MOE_DISPATCHER}'." >&2
  exit 2
fi
if [[ ! "${MOE_DEEPEP_NUM_SMS}" =~ ^[1-9][0-9]*$ ]] || (( MOE_DEEPEP_NUM_SMS % 2 != 0 )); then
  echo "ERROR: MOE_DEEPEP_NUM_SMS must be a positive even integer; got '${MOE_DEEPEP_NUM_SMS}'." >&2
  exit 2
fi
if [[ ! "${MOE_DEEPEP_TOKEN_CAPACITY}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: MOE_DEEPEP_TOKEN_CAPACITY must be a positive integer; got '${MOE_DEEPEP_TOKEN_CAPACITY}'." >&2
  exit 2
fi
case "${MOE_SHARED_EXPERT_OVERLAP,,}" in
  0|false|no|off) MOE_SHARED_EXPERT_OVERLAP=false ;;
  1|true|yes|on) MOE_SHARED_EXPERT_OVERLAP=true ;;
  *) echo "ERROR: MOE_SHARED_EXPERT_OVERLAP must be a boolean value: ${MOE_SHARED_EXPERT_OVERLAP}" >&2; exit 2 ;;
esac
case "${FSDP_FORWARD_PREFETCH,,}" in
  0|false|no|off) FSDP_FORWARD_PREFETCH=false ;;
  1|true|yes|on) FSDP_FORWARD_PREFETCH=true ;;
  *) echo "ERROR: FSDP_FORWARD_PREFETCH must be a boolean value: ${FSDP_FORWARD_PREFETCH}" >&2; exit 2 ;;
esac
case "${FSDP_BACKWARD_PREFETCH,,}" in
  0|false|no|off) FSDP_BACKWARD_PREFETCH=false ;;
  1|true|yes|on) FSDP_BACKWARD_PREFETCH=true ;;
  *) echo "ERROR: FSDP_BACKWARD_PREFETCH must be a boolean value: ${FSDP_BACKWARD_PREFETCH}" >&2; exit 2 ;;
esac
case "${FSDP_DEEPEP_STREAM_COMPAT,,}" in
  0|false|no|off) FSDP_DEEPEP_STREAM_COMPAT=false ;;
  1|true|yes|on) FSDP_DEEPEP_STREAM_COMPAT=true ;;
  *) echo "ERROR: FSDP_DEEPEP_STREAM_COMPAT must be a boolean value: ${FSDP_DEEPEP_STREAM_COMPAT}" >&2; exit 2 ;;
esac
case "${FSDP_DEEPEP_SHARED_COMM_STREAM,,}" in
  0|false|no|off) FSDP_DEEPEP_SHARED_COMM_STREAM=false ;;
  1|true|yes|on) FSDP_DEEPEP_SHARED_COMM_STREAM=true ;;
  *) echo "ERROR: FSDP_DEEPEP_SHARED_COMM_STREAM must be a boolean value: ${FSDP_DEEPEP_SHARED_COMM_STREAM}" >&2; exit 2 ;;
esac
case "${DATALOADER_USE_BACKGROUND_PREFETCHER,,}" in
  0|false|no|off) DATALOADER_USE_BACKGROUND_PREFETCHER=false ;;
  1|true|yes|on) DATALOADER_USE_BACKGROUND_PREFETCHER=true ;;
  *) echo "ERROR: DATALOADER_USE_BACKGROUND_PREFETCHER must be a boolean value: ${DATALOADER_USE_BACKGROUND_PREFETCHER}" >&2; exit 2 ;;
esac
case "${SYNC_EACH_TRAIN_STEP,,}" in
  0|false|no|off) SYNC_EACH_TRAIN_STEP=false ;;
  1|true|yes|on) SYNC_EACH_TRAIN_STEP=true ;;
  *) echo "ERROR: SYNC_EACH_TRAIN_STEP must be a boolean value: ${SYNC_EACH_TRAIN_STEP}" >&2; exit 2 ;;
esac
case "${SKIP_EMPTY_MODALITY_DUMMY,,}" in
  0|false|no|off) SKIP_EMPTY_MODALITY_DUMMY=false ;;
  1|true|yes|on) SKIP_EMPTY_MODALITY_DUMMY=true ;;
  *) echo "ERROR: SKIP_EMPTY_MODALITY_DUMMY must be a boolean value: ${SKIP_EMPTY_MODALITY_DUMMY}" >&2; exit 2 ;;
esac
case "${VISION_PATCH_EMBED_IMPLEMENTATION}" in
  conv3d|linear) ;;
  *) echo "ERROR: VISION_PATCH_EMBED_IMPLEMENTATION must be conv3d or linear: ${VISION_PATCH_EMBED_IMPLEMENTATION}" >&2; exit 2 ;;
esac
case "${CHUNK_GATED_DELTA_RULE_IMPLEMENTATION}" in
  fla|musa|musa_tilelang) ;;
  *) echo "ERROR: CHUNK_GATED_DELTA_RULE_IMPLEMENTATION must be fla, musa, or musa_tilelang: ${CHUNK_GATED_DELTA_RULE_IMPLEMENTATION}" >&2; exit 2 ;;
esac
if [[ "${MOE_SHARED_EXPERT_OVERLAP}" == true && "${MOE_DISPATCHER}" != "deepep_ace" ]]; then
  echo "ERROR: MOE_SHARED_EXPERT_OVERLAP requires MOE_DISPATCHER=deepep_ace." >&2
  exit 2
fi
if [[ "${FSDP_DEEPEP_STREAM_COMPAT}" == true ]]; then
  if [[ "${MOE_DISPATCHER}" != "deepep_ace" ]]; then
    echo "ERROR: FSDP_DEEPEP_STREAM_COMPAT requires MOE_DISPATCHER=deepep_ace." >&2
    exit 2
  fi
  if [[ "${TORCH_MUSA_FSDP2_OVERLAP_LEVEL:-0}" != 2 || "${TORCH_MUSA_FSDP2_COMM_TYPE:-0}" != 0 ]]; then
    echo "ERROR: FSDP_DEEPEP_STREAM_COMPAT is validated only with FSDP2 COMM_TYPE=0 and OVERLAP_LEVEL=2." >&2
    exit 2
  fi
elif [[ "${MOE_DISPATCHER}" == "deepep_ace" && "${TORCH_MUSA_FSDP2_OVERLAP_LEVEL:-0}" == 2 ]]; then
  echo "ERROR: raw FSDP2 level 2 deadlocks with DeepEP-ACE; set FSDP_DEEPEP_STREAM_COMPAT=true." >&2
  exit 2
fi
if [[ "${FSDP_DEEPEP_SHARED_COMM_STREAM}" == true && "${FSDP_DEEPEP_STREAM_COMPAT}" != true ]]; then
  echo "ERROR: FSDP_DEEPEP_SHARED_COMM_STREAM requires FSDP_DEEPEP_STREAM_COMPAT=true." >&2
  exit 2
fi
if [[ "${MOE_DISPATCHER}" == "deepep_ace" ]]; then
  if ! "${PYTHON_BIN}" - <<'PY'
import importlib.metadata as metadata
import inspect

from deep_ep import Buffer, EventOverlap
from deep_ep_cpp import EventHandle

required = {"use_ace", "token_num", "hidden_size", "num_topk"}
missing = required.difference(inspect.signature(Buffer).parameters)
if missing:
    raise RuntimeError(f"DeepEP Buffer is missing ACE parameters: {sorted(missing)}")
print(f"DeepEP-ACE preflight: deep_ep={metadata.version('deep_ep')}")
PY
  then
    echo "ERROR: MOE_DISPATCHER=deepep_ace requires a compatible DeepEP-ACE wheel." >&2
    exit 2
  fi
fi

for _card in "${_MUSA_CARDS[@]}"; do
  if [[ ! "${_card}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid MUSA card id '${_card}'." >&2
    exit 2
  fi

  _used_mib="$({
    mthreads-gmi --query --id "${_card}" --display MEMORY,UTILIZATION --json
  } | "${PYTHON_BIN}" -c 'import json, re, sys; d=json.load(sys.stdin); s=d["GPU"][0]["FB Memory Usage"]["Used"]; print(int(re.search(r"[0-9]+", s).group()))')"

  if (( _used_mib > MAX_EXISTING_MIB )); then
    echo "REFUSE: MUSA card ${_card} already uses ${_used_mib} MiB (limit ${MAX_EXISTING_MIB} MiB)." >&2
    echo "       Choose eight assigned/free cards with MUSA_VISIBLE_DEVICES=a,b,c,d,e,f,g,h." >&2
    exit 3
  fi
  echo "MUSA card ${_card}: ${_used_mib} MiB in use; accepted by preflight."
done

export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export TORCHRUN_PYTHON="${TORCHRUN_PYTHON:-${PYTHON_BIN}}"
export MATE_MUSA_ARCH_LIST="${MATE_MUSA_ARCH_LIST:-3.1}"
if [[ "${NPROC_PER_NODE}" != 8 ]]; then
  echo "ERROR: this launcher requires NPROC_PER_NODE=8; got '${NPROC_PER_NODE}'." >&2
  exit 2
fi

export MODEL_PATH="${MODEL_PATH:-/data/share/models/Qwen3.5-35B-A3B}"
export DATA_PATH="${DATA_PATH:-/data/share/liang.geng/fsdp_overlap_test/data/sharegpt4v_coco_full/sharegpt4v_coco_full.json}"
DATA_TYPE="${DATA_TYPE:-conversation}"
TEXT_KEYS="${TEXT_KEYS:-messages}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
TRAIN_SIZE="${TRAIN_SIZE:-52428800}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((MICRO_BATCH_SIZE * NPROC_PER_NODE))}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_3}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-50}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
# VeOmni applies train.max_steps inside each epoch, so the maximum total is
# STEPS_PER_EPOCH * NUM_TRAIN_EPOCHS.

PROFILE_ENABLE="${PROFILE_ENABLE:-0}"
PROFILE_START_STEP="${PROFILE_START_STEP:-3}"
PROFILE_END_STEP="${PROFILE_END_STEP:-6}"
PROFILE_TRACE_DIR="${PROFILE_TRACE_DIR:-${SCRIPT_DIR}/traces/qwen35a3b_image_$(date +%Y%m%d_%H%M%S)}"
PROFILE_RECORD_SHAPES="${PROFILE_RECORD_SHAPES:-false}"
PROFILE_MEMORY="${PROFILE_MEMORY:-false}"
PROFILE_STACK="${PROFILE_STACK:-false}"
PROFILE_MODULES="${PROFILE_MODULES:-false}"
PROFILE_RANK0_ONLY="${PROFILE_RANK0_ONLY:-true}"

case "${PROFILE_ENABLE,,}" in
  0|false|no|off) PROFILE_ENABLE=false ;;
  1|true|yes|on) PROFILE_ENABLE=true ;;
  *) echo "ERROR: PROFILE_ENABLE must be a boolean value: ${PROFILE_ENABLE}" >&2; exit 2 ;;
esac
if [[ ! "${PROFILE_START_STEP}" =~ ^[1-9][0-9]*$ || ! "${PROFILE_END_STEP}" =~ ^[1-9][0-9]*$ || "${PROFILE_END_STEP}" -le "${PROFILE_START_STEP}" ]]; then
  echo "ERROR: PROFILE_START_STEP/PROFILE_END_STEP must be positive and END must be greater than START." >&2
  exit 2
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: model path does not exist: ${MODEL_PATH}" >&2
  exit 2
fi
if [[ ! -f "${DATA_PATH}" ]]; then
  echo "ERROR: prepared multimodal annotation path does not exist: ${DATA_PATH}" >&2
  exit 2
fi
if [[ "${DATA_PATH}" == "${SCRIPT_DIR}"/* ]]; then
  echo "Using prepared ShareGPT4V-COCO annotation: ${DATA_PATH}"
fi
if [[ ! "${MICRO_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: MICRO_BATCH_SIZE must be a positive integer: ${MICRO_BATCH_SIZE}" >&2
  exit 2
fi
if [[ ! "${GLOBAL_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: GLOBAL_BATCH_SIZE must be a positive integer: ${GLOBAL_BATCH_SIZE}" >&2
  exit 2
fi
if [[ ! "${STEPS_PER_EPOCH}" =~ ^[1-9][0-9]*$ || ! "${NUM_TRAIN_EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: STEPS_PER_EPOCH and NUM_TRAIN_EPOCHS must be positive integers." >&2
  exit 2
fi
TOTAL_TRAIN_STEPS=$((10#${STEPS_PER_EPOCH} * 10#${NUM_TRAIN_EPOCHS}))
if [[ "${PROFILE_ENABLE}" == true ]] && (( TOTAL_TRAIN_STEPS < 10#${PROFILE_END_STEP} )); then
  echo "ERROR: the profiling window ends at step ${PROFILE_END_STEP}, but this run has at most ${TOTAL_TRAIN_STEPS} steps." >&2
  exit 2
fi
if (( GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * NPROC_PER_NODE) != 0 )); then
  echo "ERROR: GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be a multiple of MICRO_BATCH_SIZE*NPROC_PER_NODE=$((MICRO_BATCH_SIZE * NPROC_PER_NODE))." >&2
  exit 2
fi

# Keep the same private-output behavior as the reference image launcher.
_CLEANUP_OUTPUT=0
if [[ -z "${OUTPUT_DIR:-}" ]]; then
  OUTPUT_DIR="$(mktemp -d /tmp/veomni_qwen35_image_8card_smoke.XXXXXX)"
  _CLEANUP_OUTPUT=1
fi
cleanup_output() {
  if [[ "${_CLEANUP_OUTPUT}" -eq 1 && "${KEEP_OUTPUT:-0}" != "1" && -d "${OUTPUT_DIR}" ]]; then
    rm -rf -- "${OUTPUT_DIR}"
  fi
}
trap cleanup_output EXIT

echo "Starting Qwen3.5-35B-A3B eight-card image-training smoke test."
echo "  MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES}"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  DATA_PATH=${DATA_PATH}"
echo "  DATA_TYPE=${DATA_TYPE}"
echo "  TEXT_KEYS=${TEXT_KEYS}"
echo "  MAX_SEQ_LEN=${MAX_SEQ_LEN}"
echo "  MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}"
echo "  GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "  ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION}"
echo "  MOE_DISPATCHER=${MOE_DISPATCHER}"
if [[ "${MOE_DISPATCHER}" == "deepep_ace" ]]; then
  echo "  MOE_DEEPEP_NUM_SMS=${MOE_DEEPEP_NUM_SMS}"
  echo "  MOE_DEEPEP_TOKEN_CAPACITY=${MOE_DEEPEP_TOKEN_CAPACITY}"
  echo "  MOE_SHARED_EXPERT_OVERLAP=${MOE_SHARED_EXPERT_OVERLAP}"
fi
echo "  DATALOADER_USE_BACKGROUND_PREFETCHER=${DATALOADER_USE_BACKGROUND_PREFETCHER}"
echo "  SYNC_EACH_TRAIN_STEP=${SYNC_EACH_TRAIN_STEP}"
echo "  SKIP_EMPTY_MODALITY_DUMMY=${SKIP_EMPTY_MODALITY_DUMMY}"
echo "  VISION_PATCH_EMBED_IMPLEMENTATION=${VISION_PATCH_EMBED_IMPLEMENTATION}"
echo "  CHUNK_GATED_DELTA_RULE_IMPLEMENTATION=${CHUNK_GATED_DELTA_RULE_IMPLEMENTATION}"
echo "  RMS_NORM_GATED_IMPLEMENTATION=${RMS_NORM_GATED_IMPLEMENTATION}"
echo "  CAUSAL_CONV1D_IMPLEMENTATION=${CAUSAL_CONV1D_IMPLEMENTATION}"
echo "  VEOMNI_MCCL_NATIVE_AVG=${VEOMNI_MCCL_NATIVE_AVG}"
echo "  VEOMNI_MUSA_FSDP2_FOREACH_GRAD_NORM=${VEOMNI_MUSA_FSDP2_FOREACH_GRAD_NORM}"
echo "  FSDP=8 (fsdp2), EP=8, SP=1"
echo "  FSDP_FORWARD_PREFETCH=${FSDP_FORWARD_PREFETCH}"
echo "  FSDP_BACKWARD_PREFETCH=${FSDP_BACKWARD_PREFETCH}"
echo "  FSDP_DEEPEP_STREAM_COMPAT=${FSDP_DEEPEP_STREAM_COMPAT}"
echo "  FSDP_DEEPEP_SHARED_COMM_STREAM=${FSDP_DEEPEP_SHARED_COMM_STREAM}"
echo "  STEPS_PER_EPOCH=${STEPS_PER_EPOCH}"
echo "  NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "  PROFILE_ENABLE=${PROFILE_ENABLE}"
echo "  TORCH_MUSA_FSDP2_COMM_TYPE=${TORCH_MUSA_FSDP2_COMM_TYPE:-0}"
echo "  TORCH_MUSA_FSDP2_OVERLAP_LEVEL=${TORCH_MUSA_FSDP2_OVERLAP_LEVEL:-0}"
echo "  MCCL_PROTOS=${MCCL_PROTOS}"
echo "  MCCL_ALGOS=${MCCL_ALGOS}"
echo "  MCCL_BUFFSIZE=${MCCL_BUFFSIZE:-auto}"
echo "  MCCL_CTA_POLICY=${MCCL_CTA_POLICY}"
echo "  MCCL_MAX_NCHANNELS=${MCCL_MAX_NCHANNELS:-auto}"
echo "  MCCL_MIN_NCHANNELS=${MCCL_MIN_NCHANNELS:-auto}"
echo "  MUSA_BLOCK_SCHEDULE_MODE=${MUSA_BLOCK_SCHEDULE_MODE:-auto}"
echo "  MCCL_IB_GID_INDEX=${MCCL_IB_GID_INDEX:-auto}"
echo "  MCCL_NET_SHARED_BUFFERS=${MCCL_NET_SHARED_BUFFERS:-auto}"
if [[ "${PROFILE_ENABLE}" == true ]]; then
  echo "  PROFILE_TRACE_DIR=${PROFILE_TRACE_DIR}"
fi

PROFILE_ARGS=()
if [[ "${PROFILE_ENABLE}" == true ]]; then
  mkdir -p "${PROFILE_TRACE_DIR}"
  PROFILE_ARGS+=(
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
else
  PROFILE_ARGS+=(--train.profile.enable false)
fi

export TORCH_MUSA_FSDP2_COMM_TYPE="${TORCH_MUSA_FSDP2_COMM_TYPE:-0}"
# Keep the production default conservative. To combine DeepEP-ACE with level 2,
# also set FSDP_DEEPEP_STREAM_COMPAT=true; VeOmni then keeps FSDP copy-in on the
# compute stream and runs its collectives on a normal-priority stream.
export TORCH_MUSA_FSDP2_OVERLAP_LEVEL="${TORCH_MUSA_FSDP2_OVERLAP_LEVEL:-0}"
export VEOMNI_MUSA_DEEPEP_FSDP_STREAM_COMPAT="${FSDP_DEEPEP_STREAM_COMPAT}"
export VEOMNI_MUSA_DEEPEP_FSDP_SHARED_COMM_STREAM="${FSDP_DEEPEP_SHARED_COMM_STREAM}"

# The MoE VLM config supplies Qwen3.5 multimodal processing and mm_configs.
bash train.sh \
  tasks/train_vlm.py \
  configs/multimodal/qwen3_5_moe/qwen3_5_moe_vl.yaml \
  --model.model_path "${MODEL_PATH}" \
  --data.train_path "${DATA_PATH}" \
  --data.data_type "${DATA_TYPE}" \
  --data.text_keys "${TEXT_KEYS}" \
  --data.source_name sharegpt4v_sft \
  --data.train_size "${TRAIN_SIZE}" \
  --data.max_seq_len "${MAX_SEQ_LEN}" \
  --data.dataloader.pin_memory false \
  --data.dataloader.num_workers 8 \
  --data.dataloader.prefetch_factor 4 \
  --data.dataloader.persistent_workers true \
  --data.dataloader.use_background_prefetcher "${DATALOADER_USE_BACKGROUND_PREFETCHER}" \
  --train.max_steps "${STEPS_PER_EPOCH}" \
  --train.num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --train.global_batch_size "${GLOBAL_BATCH_SIZE}" \
  --train.micro_batch_size "${MICRO_BATCH_SIZE}" \
  --train.sync_each_train_step "${SYNC_EACH_TRAIN_STEP}" \
  --model.ops_implementation.attn_implementation "${ATTN_IMPLEMENTATION}" \
  --model.ops_implementation.rms_norm_implementation musa \
  --model.ops_implementation.moe_implementation fused_musa \
  --model.ops_implementation.moe_dispatcher "${MOE_DISPATCHER}" \
  --model.ops_implementation.moe_deepep_num_sms "${MOE_DEEPEP_NUM_SMS}" \
  --model.ops_implementation.moe_deepep_token_capacity "${MOE_DEEPEP_TOKEN_CAPACITY}" \
  --model.ops_implementation.moe_shared_expert_overlap "${MOE_SHARED_EXPERT_OVERLAP}" \
  --model.ops_implementation.skip_empty_modality_dummy "${SKIP_EMPTY_MODALITY_DUMMY}" \
  --model.ops_implementation.vision_patch_embed_implementation "${VISION_PATCH_EMBED_IMPLEMENTATION}" \
  --model.ops_implementation.rotary_pos_emb_implementation eager \
  --model.ops_implementation.rotary_pos_emb_vision_implementation musa \
  --model.ops_implementation.cross_entropy_loss_implementation chunk_loss \
  --model.ops_implementation.rms_norm_gated_implementation "${RMS_NORM_GATED_IMPLEMENTATION}" \
  --model.ops_implementation.causal_conv1d_implementation "${CAUSAL_CONV1D_IMPLEMENTATION}" \
  --model.ops_implementation.chunk_gated_delta_rule_implementation "${CHUNK_GATED_DELTA_RULE_IMPLEMENTATION}" \
  --model.accelerator.gradient_checkpointing.enable true \
  --model.accelerator.dp_shard_size 8 \
  --model.accelerator.ep_size 8 \
  --model.accelerator.ulysses_size 1 \
  --model.accelerator.fsdp_config.fsdp_mode fsdp2 \
  --model.accelerator.fsdp_config.forward_prefetch "${FSDP_FORWARD_PREFETCH}" \
  --model.accelerator.fsdp_config.backward_prefetch "${FSDP_BACKWARD_PREFETCH}" \
  --model.accelerator.init_device meta \
  --train.checkpoint.output_dir "${OUTPUT_DIR}" \
  --train.checkpoint.save_steps 0 \
  --train.checkpoint.save_epochs 0 \
  --train.wandb.enable false \
  "${PROFILE_ARGS[@]}"
