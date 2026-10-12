"""Fuse BF16 router rounding and FP32 ACE payload materialization."""

import ast
import functools
import hashlib
import inspect
import os
import textwrap
from collections import Counter

import torch
import triton
import triton.language as tl

from ....distributed.parallel_state import get_parallel_state
from ....ops.config.singleton import get_ops_config
from ....utils.moe_router_replay import get_active_replay


FORWARD_SHA = "fa092ef50b24a2b0d81e4830b74bdc51efb6aa45cff9f48ac69fb032f1a337ea"
counters = Counter()


@triton.jit(do_not_specialize=["N"])
def _round_payload(x, output, N, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(x + i, i < N, 0)
    rounded = value.to(tl.bfloat16).to(tl.float32)
    tl.store(output + i, rounded, i < N)


def _round(value):
    value = value.contiguous()
    output = torch.empty_like(value)
    with torch.musa.device(value.device):
        _round_payload[(triton.cdiv(value.numel(), 1024),)](value, output, N=value.numel(), B=1024)
    return output


class _RoundedPayload(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        counters["forward"] += 1
        return _round(value)

    @staticmethod
    def backward(ctx, gradient):
        counters["backward"] += 1
        if torch.is_grad_enabled():
            return gradient.to(torch.bfloat16).float()
        return _round(gradient)


def rounded_payload(value, target_dtype):
    if (
        os.environ.get("VEOMNI_MUSA_ROUTER_FP32_PAYLOAD", "0") == "1"
        and getattr(get_ops_config(), "moe_dispatcher", None) == "deepep_ace"
        and getattr(get_ops_config(), "moe_implementation", None) == "fused_musa"
        and get_parallel_state().ep_enabled
        and get_active_replay() is None
        and type(value) is torch.Tensor
        and value.device.type == "musa"
        and value.dtype == torch.float32
        and target_dtype == torch.bfloat16
        and value.numel() > 0
        and torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
    ):
        return _RoundedPayload.apply(value)
    counters["fallbacks"] += 1
    return value.to(target_dtype)


def make_candidate(original):
    source = textwrap.dedent(inspect.getsource(original))
    if hashlib.sha256(source.encode()).hexdigest() != FORWARD_SHA:
        raise RuntimeError("Unsupported router forward source for FP32 payload experiment")
    tree = ast.parse(source)

    class Replace(ast.NodeTransformer):
        count = 0

        def visit_Call(self, node):
            if ast.unparse(node) == "router_top_value.to(router_logits.dtype)":
                self.count += 1
                return ast.copy_location(
                    ast.parse("_rounded_payload(router_top_value, router_logits.dtype)").body[0].value, node
                )
            return self.generic_visit(node)

    transform = Replace()
    tree = transform.visit(tree)
    assert transform.count == 1
    namespace = dict(original.__globals__, _rounded_payload=rounded_payload)
    exec(compile(ast.fix_missing_locations(tree), inspect.getfile(original), "exec"), namespace)
    return functools.update_wrapper(namespace["forward"], original)


def install(modeling):
    cls = modeling.Qwen3_5MoeTopKRouter
    if getattr(cls.forward, "_musa_fp32_payload", False):
        return
    cls.forward = make_candidate(cls.forward)
    cls.forward._musa_fp32_payload = True
    modeling.Qwen3_5MoeExperts._veomni_qwen_rounded_payload = True


def stats():
    return dict(counters)
