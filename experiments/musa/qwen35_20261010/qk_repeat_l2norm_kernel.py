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
    x, y, rstd, eps, T,
    D: tl.constexpr, BD: tl.constexpr, NB: tl.constexpr, BT: tl.constexpr,
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
