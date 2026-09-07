# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MUSA tuned FLA adapter for Qwen3.5's chunk gated delta rule.

The tuning choices are the process-local changes from
``bench_qwen35_fla_tuned.py``. They are kept here so the ``musa`` backend has
no runtime dependency on the temporary benchmark directory or on modified
files inside the installed FLA package.
"""

from __future__ import annotations

import os
import threading
from typing import Callable

import torch


_CONFIGURE_LOCK = threading.Lock()
_CONFIGURED = False


def _autotuner(kernel):
    """Return the CachedAutotuner below any Heuristics wrappers."""
    current = kernel
    while not hasattr(current, "configs"):
        if not hasattr(current, "fn"):
            raise TypeError(f"cannot find autotuner below {type(kernel)!r}")
        current = current.fn
    return current


def _force(kernel, kwargs, num_warps, num_stages=None):
    """Freeze one Triton autotuner to the S5000-tested configuration."""
    import triton

    autotuner = _autotuner(kernel)
    config = triton.Config(kwargs, num_warps=num_warps, num_stages=num_stages)
    autotuner.configs = [config]
    autotuner.cache.clear()


def _set_tuned_configs() -> None:
    """Freeze the S5000-tested FLA Triton configurations before first launch."""
    from fla.modules.l2norm import l2norm_bwd_kernel, l2norm_fwd_kernel
    from fla.ops.common.chunk_delta_h import (
        chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64,
        chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
    )
    from fla.ops.common.chunk_o import (
        chunk_bwd_kernel_dqkwg,
        chunk_bwd_kernel_dv_local,
        chunk_fwd_kernel_o,
    )
    from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd_kernel
    from fla.ops.gated_delta_rule.wy_fast import (
        prepare_wy_repr_bwd_kernel,
        recompute_w_u_fwd_kernel,
    )
    from fla.ops.utils.cumsum import chunk_local_cumsum_scalar_kernel
    from fla.ops.utils.solve_tril import merge_16x16_to_64x64_inverse_kernel

    _force(chunk_scaled_dot_kkt_fwd_kernel, {"BK": 32}, 8, 2)
    _force(merge_16x16_to_64x64_inverse_kernel, {"DOT_PRECISION": "ieee"}, 4, 4)
    _force(chunk_gated_delta_rule_fwd_kernel_h_blockdim64, {"BV": 64}, 16, 2)
    _force(chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64, {"BV": 64}, 16, 2)
    _force(prepare_wy_repr_bwd_kernel, {}, 8, 2)
    _force(recompute_w_u_fwd_kernel, {}, 16, 2)

    # These kernels are common to both stock and tuned FLA paths.
    _force(chunk_fwd_kernel_o, {"BK": 128, "BV": 128}, 8, 3)
    _force(chunk_bwd_kernel_dv_local, {}, 8, 2)
    _force(chunk_bwd_kernel_dqkwg, {}, 8, 2)
    _force(chunk_local_cumsum_scalar_kernel, {}, 1, 2)
    _force(l2norm_fwd_kernel, {"BT": 64}, 4)
    _force(l2norm_bwd_kernel, {"BT": 16}, 2)


def _install_unfused_kkt() -> None:
    """Select FLA's existing unfused KKT+solve forward path on MUSA."""
    from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
    from fla.ops.gated_delta_rule import chunk as gdr_chunk
    from fla.ops.gated_delta_rule import chunk_fwd as gdr_chunk_fwd
    from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd
    from fla.ops.utils import prepare_chunk_indices, solve_tril

    stock = gdr_chunk_fwd.chunk_gated_delta_rule_fwd_intra

    def unfused(k, v, g=None, beta=None, cu_seqlens=None, chunk_size=64, chunk_indices=None):
        if chunk_size != 64:
            return stock(k, v, g, beta, cu_seqlens, chunk_size, chunk_indices)
        if chunk_indices is None and cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        a = chunk_scaled_dot_kkt_fwd(
            k=k,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
            output_dtype=torch.float32,
        )
        a = solve_tril(
            A=a,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            output_dtype=k.dtype,
        )
        w, u = recompute_w_u_fwd(
            k=k,
            v=v,
            beta=beta,
            A=a,
            g=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        return w, u, a

    # ChunkGatedDeltaRuleFunction resolves this symbol in its defining module.
    gdr_chunk_fwd.chunk_gated_delta_rule_fwd_intra = unfused
    gdr_chunk.chunk_gated_delta_rule_fwd_intra = unfused


def _ensure_tuned_fla() -> None:
    """Install the tuned choices exactly once in the current process."""
    global _CONFIGURED

    with _CONFIGURE_LOCK:
        if _CONFIGURED:
            return
        # Keep this backend on the Triton path even if the optional MUSA
        # forward-H bridge was enabled by the caller's environment.
        os.environ["FLA_USE_MU_KERNEL"] = "0"
        _set_tuned_configs()
        _install_unfused_kkt()
        _CONFIGURED = True


def get_musa_chunk_gated_delta_rule() -> Callable:
    """Return FLA's tuned, varlen-aware chunk gated delta-rule callable."""
    _ensure_tuned_fla()
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    return chunk_gated_delta_rule


__all__ = ["get_musa_chunk_gated_delta_rule"]
