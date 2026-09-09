import sys
import types

import pytest
import torch
import torch.nn.functional as F

from veomni.ops.kernels.moe.musa_group_gemm import musa_fused_moe_forward


def _install_fake_mate(monkeypatch: pytest.MonkeyPatch) -> None:
    def ragged_m(input_a, input_b, ragged_tokens_info, out, *, major_b_mode, **kwargs):
        del kwargs
        offset = 0
        for expert_id, count in enumerate(ragged_tokens_info.tolist()):
            end = offset + count
            expert_input = input_a[offset:end]
            expert_weight = input_b[expert_id]
            if major_b_mode == "K":
                expert_output = expert_input @ expert_weight.transpose(0, 1)
            else:
                expert_output = expert_input @ expert_weight
            out[offset:end].copy_(expert_output)
            offset = end
        return out

    def ragged_k(input_a, input_b, ragged_tokens_info, out, **kwargs):
        del kwargs
        offset = 0
        for expert_id, count in enumerate(ragged_tokens_info.tolist()):
            end = offset + count
            out[expert_id].add_(input_a[offset:end].transpose(0, 1) @ input_b[offset:end])
            offset = end
        return out

    gemm_module = types.ModuleType("mate.gemm")
    gemm_module.ragged_m_moe_gemm_16bit = ragged_m
    gemm_module.ragged_k_moe_gemm_16bit = ragged_k
    mate_module = types.ModuleType("mate")
    mate_module.gemm = gemm_module
    monkeypatch.setitem(sys.modules, "mate", mate_module)
    monkeypatch.setitem(sys.modules, "mate.gemm", gemm_module)


def _reference_moe(hidden_states, routing_weights, selected_experts, fc1_1_weight, fc1_2_weight, fc2_weight, limit):
    output = torch.zeros_like(hidden_states)
    for token_id in range(hidden_states.shape[0]):
        for slot_id in range(selected_experts.shape[1]):
            expert_id = int(selected_experts[token_id, slot_id])
            gate = F.linear(hidden_states[token_id], fc1_1_weight[expert_id])
            up = F.linear(hidden_states[token_id], fc1_2_weight[expert_id])
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
            activation = F.silu(gate) * up
            activation = activation * routing_weights[token_id, slot_id]
            output[token_id] = output[token_id] + F.linear(activation, fc2_weight[expert_id])
    return output


@pytest.mark.parametrize("merged", [False, True])
def test_musa_fused_moe_independent_autograd(monkeypatch: pytest.MonkeyPatch, merged: bool):
    _install_fake_mate(monkeypatch)
    torch.manual_seed(17)
    experts, tokens, hidden, intermediate, top_k = 3, 6, 4, 3, 2
    selected_experts = torch.tensor([[0, 1], [2, 0], [1, 2], [0, 2], [2, 1], [1, 0]], dtype=torch.int64)
    routing_weights = torch.rand(tokens, top_k, dtype=torch.float64)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    base_tensors = [
        torch.randn(tokens, hidden, dtype=torch.float64),
        torch.randn(experts, intermediate, hidden, dtype=torch.float64),
        torch.randn(experts, intermediate, hidden, dtype=torch.float64),
        torch.randn(experts, hidden, intermediate, dtype=torch.float64),
    ]
    limit = 0.7

    actual_inputs = [tensor.clone().requires_grad_() for tensor in base_tensors]
    actual_routing = routing_weights.clone().requires_grad_()
    merged_weight = torch.cat([actual_inputs[1], actual_inputs[2]], dim=1) if merged else None
    actual = musa_fused_moe_forward(
        experts,
        actual_routing,
        selected_experts,
        actual_inputs[0],
        None if merged else actual_inputs[1],
        None if merged else actual_inputs[2],
        actual_inputs[3],
        merged_weight,
        limit,
    )
    upstream = torch.randn_like(actual)
    (actual * upstream).sum().backward()

    reference_inputs = [tensor.clone().requires_grad_() for tensor in base_tensors]
    reference_routing = routing_weights.clone().requires_grad_()
    reference = _reference_moe(
        reference_inputs[0],
        reference_routing,
        selected_experts,
        reference_inputs[1],
        reference_inputs[2],
        reference_inputs[3],
        limit,
    )
    (reference * upstream).sum().backward()

    torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-10)
    for actual_tensor, reference_tensor in zip(
        [actual_routing, *actual_inputs], [reference_routing, *reference_inputs]
    ):
        torch.testing.assert_close(actual_tensor.grad, reference_tensor.grad, rtol=1e-10, atol=1e-10)
