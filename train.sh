#!/bin/bash

set -x
set -o pipefail

# Make a fresh clone runnable without requiring an editable package install.
# torch.distributed.run executes the training script from ``tasks/`` and does
# not always keep the repository root on ``sys.path``.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

NNODES=${NNODES:=1}
if command -v nvidia-smi &> /dev/null && nvidia-smi --list-gpus &> /dev/null; then
  # GPU
  if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
    NPROC_PER_NODE=${NPROC_PER_NODE:=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)}
  else
    NPROC_PER_NODE=${NPROC_PER_NODE:=$(nvidia-smi --list-gpus | wc -l)}
  fi
  export NCCL_DEBUG=WARN
elif command -v rocm-smi &> /dev/null && rocm-smi --showid &> /dev/null; then
  # AMD GPU (ROCm/HIP). Torch exposes ROCm devices through the CUDA API, and
  # RCCL reuses the NCCL_* environment variables, so most of the CUDA path
  # applies. Visibility is controlled by HIP_VISIBLE_DEVICES (falling back to
  # CUDA_VISIBLE_DEVICES, which ROCm torch also honors).
  visible_devices="${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}"
  if [[ -n "${visible_devices}" ]]; then
    NPROC_PER_NODE=${NPROC_PER_NODE:=$(echo "${visible_devices}" | tr ',' '\n' | wc -l)}
  else
    NPROC_PER_NODE=${NPROC_PER_NODE:=$(rocm-smi --showid --csv | grep -c '^card')}
  fi
  export NCCL_DEBUG=WARN
elif command -v cnmon &> /dev/null && cnmon -l 2>/dev/null | grep -q "MLU"; then
  # MLU
  if [[ -n "${MLU_VISIBLE_DEVICES}" ]]; then
    NPROC_PER_NODE=${NPROC_PER_NODE:=$(echo "${MLU_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)}
  else
    NPROC_PER_NODE=${NPROC_PER_NODE:=$(cnmon -l 2>/dev/null | grep -c "MLU")}
  fi
elif command -v mthreads-gmi &> /dev/null && mthreads-gmi --list-gpus &> /dev/null; then
  # Moore Threads MUSA. torch_musa exposes a separate ``musa`` device
  # namespace, so it must be detected before the generic NPU fallback.
  if [[ -n "${MUSA_VISIBLE_DEVICES}" ]]; then
    NPROC_PER_NODE=${NPROC_PER_NODE:=$(echo "${MUSA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)}
  else
    NPROC_PER_NODE=${NPROC_PER_NODE:=$(mthreads-gmi --list-gpus | wc -l)}
  fi
  export PYTORCH_MUSA_ALLOC_CONF="${PYTORCH_MUSA_ALLOC_CONF:-expandable_segments:True}"
  export TORCH_MCCL_AVOID_RECORD_STREAMS="${TORCH_MCCL_AVOID_RECORD_STREAMS:-1}"
  export MUSA_EXECUTION_TIMEOUT="${MUSA_EXECUTION_TIMEOUT:-3200000}"
  export ACCELERATOR_BACKEND="${ACCELERATOR_BACKEND:-musa}"
  export MCCL_PROTOS="${MCCL_PROTOS:-2}"
  export MCCL_ALGOS="${MCCL_ALGOS:-1}"
  export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
  export LD_LIBRARY_PATH="/usr/local/musa/lib:/usr/local/openmpi/lib:${LD_LIBRARY_PATH:-}"
  export PATH="/usr/local/musa/bin:/usr/local/musa/mudnn/bin:/usr/local/musa/mudnn_bench/bin:/usr/local/musa/mccl_test:/usr/local/openmpi/bin:${PATH}"
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
else
  # NPU
  if [[ -n "${ASCEND_RT_VISIBLE_DEVICES}" ]]; then
    NPROC_PER_NODE=${NPROC_PER_NODE:=$(echo "${ASCEND_RT_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)}
  else
    NPROC_PER_NODE=${NPROC_PER_NODE:=$(ls -l /dev/davinci* | grep -v "davinci_manager" | wc -l)}
  fi
  # NPU env that may optimize performance
  export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:='expandable_segments:True'}
  export MULTI_STREAM_MEMORY_REUSE=${MULTI_STREAM_MEMORY_REUSE:=2}
fi
NODE_RANK=${NODE_RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}
MASTER_PORT=${MASTER_PORT:=12345}

if [[ "$NNODES" == "1" ]]; then
  additional_args="$additional_args --standalone"
else
  additional_args="--rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
fi

# Use the active Python environment's torch.distributed launcher.  The system
# `/usr/local/bin/torchrun` in the MUSA image may be shebang-bound to
# `/usr/bin/python3`, which can silently omit the active venv's Transformers,
# torchdata, and VeOmni dependencies.
TORCHRUN_PYTHON="${TORCHRUN_PYTHON:-${PYTHON:-python}}"
"${TORCHRUN_PYTHON}" -m torch.distributed.run \
  --nnodes=$NNODES \
  --nproc-per-node=$NPROC_PER_NODE \
  --node-rank=$NODE_RANK \
  $additional_args $@ 2>&1 | tee log.txt
