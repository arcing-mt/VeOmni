# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""FLA row normalization with a two-head repeat folded into the input address.
Derived from the guarded installed l2norm_fwd_kernel. Only the input pointer
and its stride argument change; output rows and floating expressions remain.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def qk_repeat_l2norm_fwd_kernel(
    x,
    y,
    rstd,
    eps,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    BT: tl.constexpr,
    STRIDE_T: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_t = o_t < T
    m_x = m_t[:, None] & (o_d[None, :] < D)
    p_x = x + (o_t // 32)[:, None] * STRIDE_T + ((o_t % 32) // 2)[:, None] * D + o_d[None, :]
    p_y = y + o_t[:, None] * D + o_d[None, :]
    p_rstd = rstd + o_t
    b_x = tl.load(p_x, mask=m_x, other=0.0).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)
    b_y = b_x * b_rstd[:, None]

    tl.store(p_y, b_y.to(p_y.dtype.element_ty), mask=m_x)
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), mask=m_t)


@triton.jit(do_not_specialize=["T"])
def qk_joint_l2norm_fwd_kernel(
    query, key, yq, yk, rq, rk, T, BT: tl.constexpr, D: tl.constexpr, STRIDE_T: tl.constexpr
):
    which = tl.program_id(1)
    x = tl.where(which == 0, query, key)
    y = tl.where(which == 0, yq, yk)
    rstd = tl.where(which == 0, rq, rk)
    row = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    dim = tl.arange(0, D)
    valid = row < T
    offset = (row // 32)[:, None] * STRIDE_T + ((row % 32) // 2)[:, None] * D + dim[None, :]
    value = tl.load(x + offset, valid[:, None], 0.0).to(tl.float32)
    reciprocal = 1 / tl.sqrt(tl.sum(value * value, 1) + 1e-6)
    normalized = value * reciprocal[:, None]
    tl.store(y + row[:, None] * D + dim[None, :], normalized.to(y.dtype.element_ty), valid[:, None])
    tl.store(rstd + row, reciprocal, valid)


@triton.jit(do_not_specialize=["PAIRS"])
def qk_pair_l2norm_bwd_kernel(y, rstd, dy, dx, PAIRS, D: tl.constexpr, BT: tl.constexpr):
    block = tl.program_id(0).to(tl.int64)
    rows = block * BT + tl.arange(0, BT)
    dim = tl.arange(0, D)
    valid = rows < PAIRS * 2
    offsets = rows[:, None] * D + dim[None, :]
    b_y = tl.load(y + offsets, valid[:, None], 0).to(tl.float32)
    b_rstd = tl.load(rstd + rows, valid, 0).to(tl.float32)
    b_dy = tl.load(dy + offsets, valid[:, None], 0).to(tl.float32)
    # Retain the original row geometry and expression before BF16 rounding.
    b_dx = b_dy * b_rstd[:, None] - tl.sum(b_dy * b_y, 1)[:, None] * b_y * b_rstd[:, None]
    rounded = b_dx.to(dx.dtype.element_ty).to(tl.float32)
    pairs = tl.reshape(rounded, (BT // 2, 2, D))
    result = tl.sum(pairs, axis=1)
    output_rows = block * (BT // 2) + tl.arange(0, BT // 2)
    tl.store(dx + output_rows[:, None] * D + dim[None, :], result, output_rows[:, None] < PAIRS)
