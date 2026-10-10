#!/usr/bin/env bash
set -euo pipefail
artifact_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
args=()
for arg in "$@"; do
  if [[ "${arg}" == tasks/train_vlm.py ]]; then
    args+=("${artifact_root}/${TIMING_ENTRY_SCRIPT:-instrumented_train_vlm.py}")
  else
    args+=("${arg}")
  fi
done
exec python3 "${args[@]}"
