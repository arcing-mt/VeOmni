"""MUSA fused MoE forward and backward backed by MATE GroupGEMM."""

from __future__ import annotations

import torch

from ....distributed.parallel_state import get_parallel_state
from ._scatter import compute_expert_scatter_index


def validate_mate() -> None:
    try:
        from mate.gemm import ragged_k_moe_gemm_16bit, ragged_m_moe_gemm_16bit
    except (AttributeError, ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            "fused_musa requires MATE with ragged_m_moe_gemm_16bit and ragged_k_moe_gemm_16bit available"
        ) from exc
    if not callable(ragged_m_moe_gemm_16bit) or not callable(ragged_k_moe_gemm_16bit):
        raise RuntimeError("fused_musa requires callable MATE M/K-grouped GEMM interfaces")


def _musa_expert_histogram(expert_index: torch.Tensor, num_experts: int) -> torch.Tensor:
    return torch.bincount(expert_index.reshape(-1).to(torch.int64), minlength=num_experts).to(torch.int32)


def _musa_moe_scatter(hidden_states: torch.Tensor, scatter_index: torch.Tensor) -> torch.Tensor:
    repeated = hidden_states.unsqueeze(1).expand(-1, scatter_index.shape[1], -1).reshape(-1, hidden_states.shape[-1])
    scatter_index = scatter_index.reshape(-1).to(torch.int64)
    output = torch.empty_like(repeated)
    output.index_copy_(0, scatter_index, repeated)
    return output


def _musa_moe_gather(expert_output: torch.Tensor, scatter_index: torch.Tensor) -> torch.Tensor:
    num_tokens, top_k = scatter_index.shape
    scatter_index = scatter_index.reshape(-1).to(torch.int64)
    gathered = expert_output.index_select(0, scatter_index)
    return gathered.reshape(num_tokens, top_k, expert_output.shape[-1]).sum(dim=1)


def _counts_from_cumsum(cumsum: torch.Tensor) -> torch.Tensor:
    counts = torch.empty_like(cumsum, dtype=torch.int32)
    counts[0] = cumsum[0]
    counts[1:] = cumsum[1:] - cumsum[:-1]
    return counts


def musa_group_gemm_same_nk(
    input_tensor: torch.Tensor,
    right_tensor: torch.Tensor,
    cumsum_m: torch.Tensor,
    max_m: int,
    transpose_a: bool = False,
    transpose_b: bool = False,
) -> torch.Tensor:
    """Run variable-M grouped GEMM for forward and input gradients."""
    del max_m
    from mate.gemm import ragged_m_moe_gemm_16bit

    if transpose_a:
        raise NotImplementedError("MATE M-grouped GEMM requires non-transposed A")
    input_tensor = input_tensor.contiguous()
    right_tensor = right_tensor.contiguous()
    if transpose_b:
        output_width = right_tensor.shape[1]
        major_b_mode = "K"
    else:
        output_width = right_tensor.shape[2]
        major_b_mode = "N"
    output = torch.empty((input_tensor.shape[0], output_width), dtype=input_tensor.dtype, device=input_tensor.device)
    return ragged_m_moe_gemm_16bit(
        input_tensor,
        right_tensor,
        _counts_from_cumsum(cumsum_m),
        output,
        gemm_mode="per_expert",
        major_b_mode=major_b_mode,
        backend="mubin",
    )


def musa_group_gemm_same_mn(
    input_tensor: torch.Tensor,
    right_tensor: torch.Tensor,
    output: torch.Tensor,
    cumsum_k: torch.Tensor,
    max_k: int,
    transpose_a: bool = False,
    transpose_b: bool = False,
) -> torch.Tensor:
    """Run variable-K grouped GEMM for weight gradients."""
    del max_k
    from mate.gemm import ragged_k_moe_gemm_16bit

    if not transpose_a or transpose_b:
        raise NotImplementedError("MATE K-grouped GEMM supports only transposed A and non-transposed B")
    if not output.is_contiguous():
        raise ValueError("MATE K-grouped GEMM requires a contiguous output tensor")
    output.zero_()
    return ragged_k_moe_gemm_16bit(
        input_tensor.contiguous(),
        right_tensor.contiguous(),
        _counts_from_cumsum(cumsum_k),
        output,
    )


def _apply_swiglu_clamp(fc1_1_output, fc1_2_output, swiglu_limit):
    """gpt-oss / DeepSeek-V4 style clamped SwiGLU pre-activation.

    Returns ``(fc1_1_clamped, fc1_2_clamped, mask_fc1_1, mask_fc1_2)``.
    When ``swiglu_limit`` is ``None`` this is a no-op and masks are ``None``;
    callers may skip the corresponding mask multiplications in backward.

    Semantics mirror ``torch.clamp`` autograd: gradients vanish where the
    pre-clamp value falls outside the bound. ``gate`` is upper-bounded only
    (``max=limit``) so the negative tail still feeds SiLU; ``up`` is
    symmetrically clamped to ``[-limit, +limit]``.
    """
    if swiglu_limit is None:
        return fc1_1_output, fc1_2_output, None, None
    mask_fc1_1 = fc1_1_output <= swiglu_limit
    mask_fc1_2 = (fc1_2_output >= -swiglu_limit) & (fc1_2_output <= swiglu_limit)
    fc1_1_output = fc1_1_output.clamp(max=swiglu_limit)
    fc1_2_output = fc1_2_output.clamp(min=-swiglu_limit, max=swiglu_limit)
    return fc1_1_output, fc1_2_output, mask_fc1_1, mask_fc1_2


class MusaFusedMoeExpertFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        num_experts,
        gate_weights,
        expert_index,
        hidden_states,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
        swiglu_limit=None,
    ):
        splits = _musa_expert_histogram(expert_index, num_experts)

        _, scatter_index = compute_expert_scatter_index(expert_index)

        scatter_output = _musa_moe_scatter(hidden_states, scatter_index)

        cumsum_t = torch.cumsum(splits, dim=0)
        fc1_1_output = musa_group_gemm_same_nk(
            input_tensor=scatter_output,
            right_tensor=fc1_1_weight,
            cumsum_m=cumsum_t,
            max_m=scatter_output.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        fc1_2_output = musa_group_gemm_same_nk(
            input_tensor=scatter_output,
            right_tensor=fc1_2_weight,
            cumsum_m=cumsum_t,
            max_m=scatter_output.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        fc1_1_output, fc1_2_output, mask_fc1_1, mask_fc1_2 = _apply_swiglu_clamp(
            fc1_1_output, fc1_2_output, swiglu_limit
        )

        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)

        fc1_activation = fc1_1_activation * fc1_2_output

        reshaped_gate_weight = gate_weights.reshape(-1, 1)
        scattered_gate_weight = torch.empty_like(reshaped_gate_weight)
        scattered_gate_weight[scatter_index.flatten()] = reshaped_gate_weight

        fc1_weighted_output = fc1_activation * scattered_gate_weight

        fc2_output = musa_group_gemm_same_nk(
            input_tensor=fc1_weighted_output,
            right_tensor=fc2_weight,
            cumsum_m=cumsum_t,
            max_m=scatter_output.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        expert_output = _musa_moe_gather(fc2_output, scatter_index)

        output = expert_output.reshape(hidden_states.shape)

        ctx.num_experts = num_experts
        ctx.swiglu_limit = swiglu_limit
        ctx.save_for_backward(
            gate_weights,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            hidden_states,
            scatter_index,
            scatter_output,
            cumsum_t,
            fc1_1_output,
            fc1_2_output,
            fc1_activation,
            scattered_gate_weight,
            fc1_weighted_output,
            mask_fc1_1 if mask_fc1_1 is not None else torch.empty(0, device=hidden_states.device),
            mask_fc1_2 if mask_fc1_2 is not None else torch.empty(0, device=hidden_states.device),
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            gate_weights,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            hidden_states,
            scatter_index,
            scatter_output,
            cumsum_t,
            fc1_1_output,
            fc1_2_output,
            fc1_activation,
            scattered_gate_weight,
            fc1_weighted_output,
            mask_fc1_1,
            mask_fc1_2,
        ) = ctx.saved_tensors
        swiglu_limit = ctx.swiglu_limit
        hidden_dim = grad_output.shape[-1]
        grad_output = grad_output.view(-1, hidden_dim)

        grad_fc2_output = _musa_moe_scatter(grad_output, scatter_index)
        num_scattered_tokens = grad_fc2_output.shape[0]

        grad_fc1_weighted_output = musa_group_gemm_same_nk(
            input_tensor=grad_fc2_output,
            right_tensor=fc2_weight,
            cumsum_m=cumsum_t,
            max_m=num_scattered_tokens,
            transpose_b=False,
        )

        grad_fc2_weight = None
        if fc2_weight.requires_grad:
            grad_fc2_weight = torch.empty_like(fc2_weight)
            musa_group_gemm_same_mn(
                input_tensor=grad_fc2_output,
                right_tensor=fc1_weighted_output,
                output=grad_fc2_weight,
                cumsum_k=cumsum_t,
                max_k=num_scattered_tokens,
                transpose_a=True,
                transpose_b=False,
            )

        grad_fc1_activation = grad_fc1_weighted_output * scattered_gate_weight

        grad_scattered_gate_weight = torch.sum(fc1_activation * grad_fc1_weighted_output, dim=-1)
        grad_gate_weight = grad_scattered_gate_weight[scatter_index.flatten()]
        grad_gate_weight = grad_gate_weight.reshape(gate_weights.shape)

        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)

        grad_fc1_1_activation = grad_fc1_activation * fc1_2_output
        grad_fc1_2_output = fc1_1_activation * grad_fc1_activation

        if swiglu_limit is not None:
            grad_fc1_2_output.masked_fill_(~mask_fc1_2, 0)

        grad_scatter_output_2 = musa_group_gemm_same_nk(
            input_tensor=grad_fc1_2_output,
            right_tensor=fc1_2_weight,
            cumsum_m=cumsum_t,
            max_m=num_scattered_tokens,
            transpose_b=False,
        )

        grad_fc1_2_weight = None
        if fc1_2_weight.requires_grad:
            grad_fc1_2_weight = torch.empty_like(fc1_2_weight)
            musa_group_gemm_same_mn(
                input_tensor=grad_fc1_2_output,
                right_tensor=scatter_output,
                output=grad_fc1_2_weight,
                cumsum_k=cumsum_t,
                max_k=num_scattered_tokens,
                transpose_a=True,
                transpose_b=False,
            )

        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)
        if swiglu_limit is not None:
            grad_fc1_1_output.masked_fill_(~mask_fc1_1, 0)

        grad_scatter_output_1 = musa_group_gemm_same_nk(
            input_tensor=grad_fc1_1_output,
            right_tensor=fc1_1_weight,
            cumsum_m=cumsum_t,
            max_m=num_scattered_tokens,
            transpose_b=False,
        )

        grad_fc1_1_weight = None
        if fc1_1_weight.requires_grad:
            grad_fc1_1_weight = torch.empty_like(fc1_1_weight)
            musa_group_gemm_same_mn(
                input_tensor=grad_fc1_1_output,
                right_tensor=scatter_output,
                output=grad_fc1_1_weight,
                cumsum_k=cumsum_t,
                max_k=num_scattered_tokens,
                transpose_a=True,
                transpose_b=False,
            )

        grad_scatter_output = grad_scatter_output_1 + grad_scatter_output_2
        grad_hidden_states = _musa_moe_gather(grad_scatter_output, scatter_index)

        grad_hidden_states = grad_hidden_states.reshape(hidden_states.shape)

        return (
            None,
            grad_gate_weight,
            None,
            grad_hidden_states,
            grad_fc1_1_weight,
            grad_fc1_2_weight,
            grad_fc2_weight,
            None,
        )


class MergedFc1MusaFusedMoeExpertFunction(torch.autograd.Function):
    """Fused MoE autograd function that natively accepts a merged fc1_1_2 weight [E, 2I, H].

    Uses a single musa_group_gemm_same_nk call for fc1 instead of two separate calls,
    avoiding the split+contiguous copy when the caller already has merged weights.
    """

    @staticmethod
    def forward(
        ctx,
        num_experts,
        gate_weights,
        expert_index,
        hidden_states,
        fc1_1_2_weight,
        fc2_weight,
        swiglu_limit=None,
    ):
        splits = _musa_expert_histogram(expert_index, num_experts)
        _, scatter_index = compute_expert_scatter_index(expert_index)
        scatter_output = _musa_moe_scatter(hidden_states, scatter_index)

        cumsum_t = torch.cumsum(splits, dim=0)

        fc1_output = musa_group_gemm_same_nk(
            input_tensor=scatter_output,
            right_tensor=fc1_1_2_weight,
            cumsum_m=cumsum_t,
            max_m=scatter_output.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        fc1_1_output, fc1_2_output = fc1_output.chunk(2, dim=-1)

        fc1_1_output, fc1_2_output, mask_fc1_1, mask_fc1_2 = _apply_swiglu_clamp(
            fc1_1_output, fc1_2_output, swiglu_limit
        )

        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_activation = fc1_1_activation * fc1_2_output

        reshaped_gate_weight = gate_weights.reshape(-1, 1)
        scattered_gate_weight = torch.empty_like(reshaped_gate_weight)
        scattered_gate_weight[scatter_index.flatten()] = reshaped_gate_weight

        fc1_weighted_output = fc1_activation * scattered_gate_weight

        fc2_output = musa_group_gemm_same_nk(
            input_tensor=fc1_weighted_output,
            right_tensor=fc2_weight,
            cumsum_m=cumsum_t,
            max_m=scatter_output.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        expert_output = _musa_moe_gather(fc2_output, scatter_index)
        output = expert_output.reshape(hidden_states.shape)

        ctx.num_experts = num_experts
        ctx.swiglu_limit = swiglu_limit
        ctx.save_for_backward(
            gate_weights,
            fc1_1_2_weight,
            fc2_weight,
            hidden_states,
            scatter_index,
            scatter_output,
            cumsum_t,
            fc1_1_output,
            fc1_2_output,
            fc1_activation,
            scattered_gate_weight,
            fc1_weighted_output,
            mask_fc1_1 if mask_fc1_1 is not None else torch.empty(0, device=hidden_states.device),
            mask_fc1_2 if mask_fc1_2 is not None else torch.empty(0, device=hidden_states.device),
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            gate_weights,
            fc1_1_2_weight,
            fc2_weight,
            hidden_states,
            scatter_index,
            scatter_output,
            cumsum_t,
            fc1_1_output,
            fc1_2_output,
            fc1_activation,
            scattered_gate_weight,
            fc1_weighted_output,
            mask_fc1_1,
            mask_fc1_2,
        ) = ctx.saved_tensors
        swiglu_limit = ctx.swiglu_limit
        hidden_dim = grad_output.shape[-1]
        grad_output = grad_output.view(-1, hidden_dim)

        grad_fc2_output = _musa_moe_scatter(grad_output, scatter_index)
        num_scattered_tokens = grad_fc2_output.shape[0]

        grad_fc1_weighted_output = musa_group_gemm_same_nk(
            input_tensor=grad_fc2_output,
            right_tensor=fc2_weight,
            cumsum_m=cumsum_t,
            max_m=num_scattered_tokens,
            transpose_b=False,
        )

        grad_fc2_weight = None
        if fc2_weight.requires_grad:
            grad_fc2_weight = torch.empty_like(fc2_weight)
            musa_group_gemm_same_mn(
                input_tensor=grad_fc2_output,
                right_tensor=fc1_weighted_output,
                output=grad_fc2_weight,
                cumsum_k=cumsum_t,
                max_k=num_scattered_tokens,
                transpose_a=True,
                transpose_b=False,
            )

        grad_fc1_activation = grad_fc1_weighted_output * scattered_gate_weight

        grad_scattered_gate_weight = torch.sum(fc1_activation * grad_fc1_weighted_output, dim=-1)
        grad_gate_weight = grad_scattered_gate_weight[scatter_index.flatten()]
        grad_gate_weight = grad_gate_weight.reshape(gate_weights.shape)

        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)

        grad_fc1_1_activation = grad_fc1_activation * fc1_2_output
        grad_fc1_2_output = fc1_1_activation * grad_fc1_activation

        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)

        if swiglu_limit is not None:
            grad_fc1_1_output.masked_fill_(~mask_fc1_1, 0)
            grad_fc1_2_output.masked_fill_(~mask_fc1_2, 0)

        grad_fc1_output = torch.cat([grad_fc1_1_output, grad_fc1_2_output], dim=-1)

        grad_scatter_output = musa_group_gemm_same_nk(
            input_tensor=grad_fc1_output,
            right_tensor=fc1_1_2_weight,
            cumsum_m=cumsum_t,
            max_m=num_scattered_tokens,
            transpose_b=False,
        )

        grad_fc1_1_2_weight = None
        if fc1_1_2_weight.requires_grad:
            grad_fc1_1_2_weight = torch.empty_like(fc1_1_2_weight)
            musa_group_gemm_same_mn(
                input_tensor=grad_fc1_output,
                right_tensor=scatter_output,
                output=grad_fc1_1_2_weight,
                cumsum_k=cumsum_t,
                max_k=num_scattered_tokens,
                transpose_a=True,
                transpose_b=False,
            )

        grad_hidden_states = _musa_moe_gather(grad_scatter_output, scatter_index)
        grad_hidden_states = grad_hidden_states.reshape(hidden_states.shape)

        return (
            None,
            grad_gate_weight,
            None,
            grad_hidden_states,
            grad_fc1_1_2_weight,
            grad_fc2_weight,
            None,
        )


class MusaEPGroupGemm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        permute_tokens,
        cumsum,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
        swiglu_limit=None,
    ):

        fc1_1_output = musa_group_gemm_same_nk(
            input_tensor=permute_tokens,
            right_tensor=fc1_1_weight,
            cumsum_m=cumsum,
            max_m=permute_tokens.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        fc1_2_output = musa_group_gemm_same_nk(
            input_tensor=permute_tokens,
            right_tensor=fc1_2_weight,
            cumsum_m=cumsum,
            max_m=permute_tokens.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        fc1_1_output, fc1_2_output, mask_fc1_1, mask_fc1_2 = _apply_swiglu_clamp(
            fc1_1_output, fc1_2_output, swiglu_limit
        )

        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)

        fc1_output = fc1_1_activation * fc1_2_output

        fc2_output = musa_group_gemm_same_nk(
            input_tensor=fc1_output,
            right_tensor=fc2_weight,
            cumsum_m=cumsum,
            max_m=permute_tokens.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        ctx.swiglu_limit = swiglu_limit
        ctx.save_for_backward(
            permute_tokens,
            cumsum,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
            mask_fc1_1 if mask_fc1_1 is not None else torch.empty(0, device=permute_tokens.device),
            mask_fc1_2 if mask_fc1_2 is not None else torch.empty(0, device=permute_tokens.device),
        )

        return fc2_output

    @staticmethod
    def backward(ctx, grad_output):
        (
            permute_tokens,
            cumsum,
            fc1_1_weight,
            fc1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
            mask_fc1_1,
            mask_fc1_2,
        ) = ctx.saved_tensors
        swiglu_limit = ctx.swiglu_limit

        grad_fc1_output = musa_group_gemm_same_nk(
            input_tensor=grad_output,
            right_tensor=fc2_weight,
            cumsum_m=cumsum,
            max_m=grad_output.shape[0],
            transpose_b=False,
        )

        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_output = fc1_1_activation * fc1_2_output

        grad_fc2_weight = None
        if fc2_weight.requires_grad:
            grad_fc2_weight = torch.empty_like(fc2_weight)
            musa_group_gemm_same_mn(
                input_tensor=grad_output,
                right_tensor=fc1_output,
                output=grad_fc2_weight,
                cumsum_k=cumsum,
                max_k=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        grad_fc1_2_output = fc1_1_activation * grad_fc1_output
        grad_fc1_1_activation = grad_fc1_output * fc1_2_output

        if swiglu_limit is not None:
            grad_fc1_2_output.masked_fill_(~mask_fc1_2, 0)

        grad_scatter_output_2 = musa_group_gemm_same_nk(
            input_tensor=grad_fc1_2_output,
            right_tensor=fc1_2_weight,
            cumsum_m=cumsum,
            max_m=grad_output.shape[0],
            transpose_b=False,
        )

        grad_fc1_2_weight = None
        if fc1_2_weight.requires_grad:
            grad_fc1_2_weight = torch.empty_like(fc1_2_weight)
            musa_group_gemm_same_mn(
                input_tensor=grad_fc1_2_output,
                right_tensor=permute_tokens,
                output=grad_fc1_2_weight,
                cumsum_k=cumsum,
                max_k=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)
        if swiglu_limit is not None:
            grad_fc1_1_output.masked_fill_(~mask_fc1_1, 0)

        grad_scatter_output_1 = musa_group_gemm_same_nk(
            input_tensor=grad_fc1_1_output,
            right_tensor=fc1_1_weight,
            cumsum_m=cumsum,
            max_m=grad_output.shape[0],
            transpose_b=False,
        )

        grad_fc1_1_weight = None
        if fc1_1_weight.requires_grad:
            grad_fc1_1_weight = torch.empty_like(fc1_1_weight)
            musa_group_gemm_same_mn(
                input_tensor=grad_fc1_1_output,
                right_tensor=permute_tokens,
                output=grad_fc1_1_weight,
                cumsum_k=cumsum,
                max_k=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        grad_permute_tokens = grad_scatter_output_1 + grad_scatter_output_2

        return (
            grad_permute_tokens,
            None,
            grad_fc1_1_weight,
            grad_fc1_2_weight,
            grad_fc2_weight,
            None,
        )


class MusaEPMergedFc1GroupGemm(torch.autograd.Function):
    """EP autograd function that accepts a merged fc1_1_2 weight [E, 2I, H].

    Uses a single musa_group_gemm_same_nk call for fc1 instead of two separate calls.
    """

    @staticmethod
    def forward(
        ctx,
        permute_tokens,
        cumsum,
        fc1_1_2_weight,
        fc2_weight,
        swiglu_limit=None,
    ):
        assert fc1_1_2_weight.shape[1] % 2 == 0, (
            f"Merged fc1_1_2_weight dim 1 must be even, got {fc1_1_2_weight.shape[1]}"
        )

        fc1_output = musa_group_gemm_same_nk(
            input_tensor=permute_tokens,
            right_tensor=fc1_1_2_weight,
            cumsum_m=cumsum,
            max_m=permute_tokens.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        fc1_1_output, fc1_2_output = fc1_output.chunk(2, dim=-1)

        fc1_1_output, fc1_2_output, mask_fc1_1, mask_fc1_2 = _apply_swiglu_clamp(
            fc1_1_output, fc1_2_output, swiglu_limit
        )

        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)

        fc1_result = fc1_1_activation * fc1_2_output

        fc2_output = musa_group_gemm_same_nk(
            input_tensor=fc1_result,
            right_tensor=fc2_weight,
            cumsum_m=cumsum,
            max_m=permute_tokens.shape[0],
            transpose_a=False,
            transpose_b=True,
        )

        ctx.swiglu_limit = swiglu_limit
        ctx.save_for_backward(
            permute_tokens,
            cumsum,
            fc1_1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
            mask_fc1_1 if mask_fc1_1 is not None else torch.empty(0, device=permute_tokens.device),
            mask_fc1_2 if mask_fc1_2 is not None else torch.empty(0, device=permute_tokens.device),
        )

        return fc2_output

    @staticmethod
    def backward(ctx, grad_output):
        (
            permute_tokens,
            cumsum,
            fc1_1_2_weight,
            fc2_weight,
            fc1_1_output,
            fc1_2_output,
            mask_fc1_1,
            mask_fc1_2,
        ) = ctx.saved_tensors
        swiglu_limit = ctx.swiglu_limit

        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_result = fc1_1_activation * fc1_2_output

        grad_fc1_result = musa_group_gemm_same_nk(
            input_tensor=grad_output,
            right_tensor=fc2_weight,
            cumsum_m=cumsum,
            max_m=grad_output.shape[0],
            transpose_b=False,
        )

        grad_fc2_weight = None
        if fc2_weight.requires_grad:
            grad_fc2_weight = torch.empty_like(fc2_weight)
            musa_group_gemm_same_mn(
                input_tensor=grad_output,
                right_tensor=fc1_result,
                output=grad_fc2_weight,
                cumsum_k=cumsum,
                max_k=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        grad_fc1_2_output = fc1_1_activation * grad_fc1_result
        grad_fc1_1_activation = grad_fc1_result * fc1_2_output
        grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)

        if swiglu_limit is not None:
            grad_fc1_1_output.masked_fill_(~mask_fc1_1, 0)
            grad_fc1_2_output.masked_fill_(~mask_fc1_2, 0)

        grad_fc1_output = torch.cat([grad_fc1_1_output, grad_fc1_2_output], dim=-1)

        grad_permute_tokens = musa_group_gemm_same_nk(
            input_tensor=grad_fc1_output,
            right_tensor=fc1_1_2_weight,
            cumsum_m=cumsum,
            max_m=grad_output.shape[0],
            transpose_b=False,
        )

        grad_fc1_1_2_weight = None
        if fc1_1_2_weight.requires_grad:
            grad_fc1_1_2_weight = torch.empty_like(fc1_1_2_weight)
            musa_group_gemm_same_mn(
                input_tensor=grad_fc1_output,
                right_tensor=permute_tokens,
                output=grad_fc1_1_2_weight,
                cumsum_k=cumsum,
                max_k=grad_output.shape[0],
                transpose_a=True,
                transpose_b=False,
            )

        return (
            grad_permute_tokens,
            None,
            grad_fc1_1_2_weight,
            grad_fc2_weight,
            None,
        )


def musa_fused_moe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor | None,
    fc1_2_weight: torch.Tensor | None,
    fc2_weight: torch.Tensor,
    fc1_1_2_weight: torch.Tensor | None = None,
    swiglu_limit: float | None = None,
):
    """MATE grouped-GEMM fused MoE forward pass.

    Accepts either split fc1 weights (fc1_1_weight, fc1_2_weight) or a merged
    fc1_1_2_weight tensor.

    - Non-EP path: dispatches to ``MergedFc1MusaFusedMoeExpertFunction`` when
      merged weights are provided, or ``MusaFusedMoeExpertFunction`` when split
      weights are provided.  No format conversion is performed.
    - EP path: always resolves to split format for ``MusaEPGroupGemm``.

    ``swiglu_limit``: gpt-oss / DeepSeek-V4 style clamp on the SwiGLU
    pre-activations (``gate.clamp(max=L)``, ``up.clamp(min=-L, max=L)``).
    ``None`` disables the clamp (default, zero overhead — used by every legacy
    MoE model).
    """
    if get_parallel_state().ep_enabled:
        from ....distributed.moe import dispatch_to_ep_class

        if fc1_1_2_weight is not None:
            if fc1_1_weight is not None or fc1_2_weight is not None:
                raise ValueError("Provide either split fc1 weights or merged fc1_1_2_weight, not both.")
            final_hidden_states = dispatch_to_ep_class(
                MusaEPMergedFc1GroupGemm,
                num_experts,
                routing_weights,
                selected_experts,
                hidden_states,
                fc1_1_2_weight,
                fc2_weight,
                swiglu_limit,
            )
        else:
            if fc1_1_weight is None or fc1_2_weight is None:
                raise ValueError("EP requires split fc1 weights (fc1_1_weight and fc1_2_weight).")
            final_hidden_states = dispatch_to_ep_class(
                MusaEPGroupGemm,
                num_experts,
                routing_weights,
                selected_experts,
                hidden_states,
                fc1_1_weight,
                fc1_2_weight,
                fc2_weight,
                swiglu_limit,
            )
    else:
        if fc1_1_2_weight is not None:
            if fc1_1_weight is not None or fc1_2_weight is not None:
                raise ValueError("Provide either split fc1 weights or merged fc1_1_2_weight, not both.")
            final_hidden_states = MergedFc1MusaFusedMoeExpertFunction.apply(
                num_experts,
                routing_weights,
                selected_experts,
                hidden_states,
                fc1_1_2_weight,
                fc2_weight,
                swiglu_limit,
            )
        else:
            if fc1_1_weight is None or fc1_2_weight is None:
                raise ValueError("Split fc1 mode requires both fc1_1_weight and fc1_2_weight.")
            final_hidden_states = MusaFusedMoeExpertFunction.apply(
                num_experts,
                routing_weights,
                selected_experts,
                hidden_states,
                fc1_1_weight,
                fc1_2_weight,
                fc2_weight,
                swiglu_limit,
            )
    return final_hidden_states
