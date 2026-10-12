"""Keep qualified router payloads as ordinary tensors through FSDP hooks."""

import os
from collections import Counter
from dataclasses import replace

import torch


counters = Counter()


def qualified_experts(module):
    if os.environ.get("VEOMNI_MUSA_ROUTER_FP32_PAYLOAD", "0") != "1" or not getattr(
        module, "_veomni_qwen_rounded_payload", False
    ):
        return False
    from ....distributed.parallel_state import get_parallel_state
    from ...config.singleton import get_ops_config

    config = get_ops_config()
    return (
        get_parallel_state().ep_enabled
        and getattr(config, "moe_dispatcher", None) == "deepep_ace"
        and getattr(config, "moe_implementation", None) == "fused_musa"
    )


def expert_fsdp_kwargs(module, kwargs):
    """Change only input casting; retain the parameter/reduction/output policy."""
    policy = kwargs.get("mp_policy")
    if (
        not qualified_experts(module)
        or policy is None
        or policy.param_dtype != torch.bfloat16
        or not policy.cast_forward_inputs
    ):
        return kwargs
    result = dict(kwargs, mp_policy=replace(policy, cast_forward_inputs=False))
    # The adapter preserves the old cast for hidden states and fallback weights.
    module._veomni_payload_input_dtype = policy.param_dtype
    counters["fsdp_input_policy"] += 1
    return result


def stats():
    return dict(counters)
