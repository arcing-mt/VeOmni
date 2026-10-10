#!/usr/bin/env bash
# Measure the validated workload; optional profiler controls stay explicit.
set -euo pipefail
export TZ=Asia/Shanghai
execution_root=/data/share/liang.geng/fsdp_overlap_test
cd "${execution_root}/VeOmni"
# The host root filesystem is full; keep this run's writable caches on /data.
cache_root="${execution_root}/cache/qwen35_resume_20261009"
export HF_HOME="${cache_root}/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
export XDG_CACHE_HOME="${cache_root}/xdg"
export TMPDIR="${cache_root}/tmp"
export TRITON_CACHE_DIR="${cache_root}/triton"
export TILELANG_CACHE_DIR="${cache_root}/tilelang"
export TILELANG_TMP_DIR="${TILELANG_CACHE_DIR}/tmp"
export TORCH_EXTENSIONS_DIR="${cache_root}/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="${cache_root}/torchinductor"
export MATE_WORKSPACE_BASE="${cache_root}/mate_workspace"
export MATE_MUBIN_DIR="${cache_root}/mate_mubin"
mkdir -p "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${XDG_CACHE_HOME}" "${TMPDIR}" \
  "${TRITON_CACHE_DIR}" "${TILELANG_TMP_DIR}" "${TORCH_EXTENSIONS_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" "${MATE_WORKSPACE_BASE}" "${MATE_MUBIN_DIR}"
training_tag="$(date +%Y%m%d_%H%M%S)"
training_log_dir="${execution_root}/logs/qwen35_optimize_${EXPERIMENT_NAME:-baseline}_${training_tag}"
mkdir -p "${training_log_dir}"
printf 'Training logs: %s\n' "${training_log_dir}"
export STEP_TIMING_DIR="${training_log_dir}"
export TORCHRUN_PYTHON="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/python_with_timing.sh"
mthreads-gmi > "${training_log_dir}/gpu_before.log"
printf 'Cache root: %s\n' "${cache_root}"
df -h / "${cache_root}" > "${training_log_dir}/disk_before.log"
export STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-50}" NUM_TRAIN_EPOCHS=1
export PROFILE_ENABLE="${PROFILE_ENABLE:-false}" KEEP_OUTPUT=1
export PROFILE_START_STEP="${PROFILE_START_STEP:-15}" PROFILE_END_STEP="${PROFILE_END_STEP:-18}"
export PROFILE_TRACE_DIR="${training_log_dir}/traces"
export PROFILE_RECORD_SHAPES="${PROFILE_RECORD_SHAPES:-false}" PROFILE_MEMORY=false PROFILE_STACK=false PROFILE_MODULES=false
export PROFILE_RANK0_ONLY="${PROFILE_RANK0_ONLY:-false}"
sha256sum run_qwen35_35b_a3b_8card_train.sh veomni/distributed/moe/deepep_ace.py > "${training_log_dir}/runtime_source.sha256"
python3 -m pip show torch torch_musa torch-kernels flash-linear-attention tilelang-musa > "${training_log_dir}/versions.log"
export OUTPUT_DIR="${execution_root}/outputs/qwen35_optimize_${EXPERIMENT_NAME:-baseline}_${training_tag}"
set +e
bash run_qwen35_35b_a3b_8card_train.sh 2>&1 | tee "${training_log_dir}/training.log"
training_status=${PIPESTATUS[0]}
set -e
printf '%s\n' "${training_status}" > "${training_log_dir}/training.exitcode"
mthreads-gmi > "${training_log_dir}/gpu_after.log"
printf 'Training exit=%s logs=%s\n' "${training_status}" "${training_log_dir}"
exit "${training_status}"
