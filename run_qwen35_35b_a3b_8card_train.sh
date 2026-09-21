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

export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a _MUSA_CARDS <<< "${MUSA_VISIBLE_DEVICES}"
if [[ "${#_MUSA_CARDS[@]}" -ne 8 ]]; then
  echo "ERROR: MUSA_VISIBLE_DEVICES must contain exactly eight cards for this launcher; got '${MUSA_VISIBLE_DEVICES}'." >&2
  exit 2
fi

# Refuse every non-zero allocation by default. Override only with an explicit
# assignment from the card owner.
MAX_EXISTING_MIB="${MAX_EXISTING_MIB:-0}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-/usr/bin/python}}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python interpreter does not exist or is not executable: ${PYTHON_BIN}" >&2
  exit 2
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

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the Qwen3.5-35B-A3B checkpoint directory}"
DATA_PATH="${DATA_PATH:?Set DATA_PATH to the prepared ShareGPT4V-COCO annotation file}"
DATA_TYPE="${DATA_TYPE:-conversation}"
TEXT_KEYS="${TEXT_KEYS:-messages}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
TRAIN_SIZE="${TRAIN_SIZE:-52428800}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((MICRO_BATCH_SIZE * NPROC_PER_NODE))}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_3}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-20}"
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
echo "  FSDP=8 (fsdp2), EP=8, SP=1"
echo "  STEPS_PER_EPOCH=${STEPS_PER_EPOCH}"
echo "  NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "  PROFILE_ENABLE=${PROFILE_ENABLE}"
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

# export TORCH_MUSA_FSDP2_COMM_TYPE=1
# export TORCH_MUSA_FSDP2_OVERLAP_LEVEL=2
# export TORCH_MUSA_FSDP2_MEMORY_EFFICIENT_OVERLAP=1
# export TORCH_MUSA_FSDP2_MEMORY_EFFICIENT_REDUCE_OUTPUT=1
export MCCL_CTA_POLICY=2

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
  --train.max_steps "${STEPS_PER_EPOCH}" \
  --train.num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --train.global_batch_size "${GLOBAL_BATCH_SIZE}" \
  --train.micro_batch_size "${MICRO_BATCH_SIZE}" \
  --model.ops_implementation.attn_implementation "${ATTN_IMPLEMENTATION}" \
  --model.ops_implementation.rms_norm_implementation musa \
  --model.ops_implementation.moe_implementation fused_musa \
  --model.ops_implementation.rotary_pos_emb_implementation eager \
  --model.ops_implementation.rotary_pos_emb_vision_implementation musa \
  --model.ops_implementation.cross_entropy_loss_implementation chunk_loss \
  --model.ops_implementation.chunk_gated_delta_rule_implementation musa \
  --train.gradient_checkpointing.enable true \
  --train.accelerator.dp_shard_size 8 \
  --train.accelerator.ep_size 8 \
  --train.accelerator.ulysses_size 1 \
  --train.accelerator.fsdp_config.fsdp_mode fsdp2 \
  --train.init_device meta \
  --train.checkpoint.output_dir "${OUTPUT_DIR}" \
  --train.checkpoint.save_steps 0 \
  --train.checkpoint.save_epochs 0 \
  --train.wandb.enable false \
  "${PROFILE_ARGS[@]}"
