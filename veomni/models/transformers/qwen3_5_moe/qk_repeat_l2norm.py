"""Guarded MUSA preentry-copy kernel for the Qwen3.5-MoE GDN runtime.

Preserve native BF16 32-head L2norm backward, followed by the original
two-head sum. Normalizing 16 heads before the gradient sum would change the
rounding order and is deliberately not implemented here.
"""

import hashlib
import importlib
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
import triton

from .qk_repeat_l2norm_kernel import qk_joint_l2norm_fwd_kernel, qk_pair_l2norm_bwd_kernel, qk_repeat_l2norm_fwd_kernel


_SHA = "5160cc4ff716c9975c655b81fd0e1bae700501dff3b4d6b045680a93185a1e99"
_CACHE_SHA = "a64b09ffd3f51aed547ad72559dee94cb9aeafd2a5f02c6c8e60a6e020147a9c"
_native = None
_cache = None
stats = Counter()


def initialize():
    global _native, _cache
    if _native is not None:
        return
    module = importlib.import_module("fla.modules.l2norm")
    cache = importlib.import_module("fla.ops.utils.cache")
    for target, expected in ((module, _SHA), (cache, _CACHE_SHA)):
        actual = hashlib.sha256(Path(target.__file__).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Unsupported FLA source: {actual}")
    _cache = cache
    _native = module


def native(x, eps=1e-6):
    initialize()
    return _native.l2norm(x.contiguous().repeat_interleave(2, dim=2), eps)


def eligible(x, eps):
    return (
        type(x) is torch.Tensor
        and x.device.type == "musa"
        and x.dtype == torch.bfloat16
        and x.ndim == 4
        and x.shape[0] == 1
        and x.shape[1] > 0
        and tuple(x.shape[2:]) == (16, 128)
        and tuple(x.stride()[1:]) == (8192, 128, 1)
        and eps == 1e-6
        and torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
    )


def native_config(x):
    """Read the original autotuner's actual choice; never tune or mutate it.
    The guarded key builder uses scalar D/NB followed by x/y/rstd dtypes in
    native argument order. rstd here is metadata only, with no GPU allocation.
    On a cold key, call the complete native path first so its normal autotuner
    chooses and caches the configuration. ALWAYS reload mode stays native.
    """
    kernel = _native.l2norm_fwd_kernel
    rows = x.shape[1] * 32
    arguments = dict(
        x=x,
        y=x,
        rstd=SimpleNamespace(dtype=torch.float32),
        eps=1e-6,
        T=rows,
        D=128,
        BD=128,
        NB=triton.cdiv(rows, 2048 * 32),
    )
    key = _cache.AutotuneKey.build(kernel.arg_names, kernel.keys, (), arguments)
    if len(kernel.configs) == 1:
        config = kernel.configs[0]
    elif kernel.should_check_fla_cache(key):
        return None
    else:
        config = kernel.cache.get(key.autotune_key)
    if config is None or config.pre_hook is not None or set(config.kwargs) != {"BT"}:
        return None
    if config.kwargs["BT"] not in (8, 16, 32, 64, 128):
        return None
    return config


def native_backward_config(y, dy):
    """Keep the native row reduction geometry, including cold-cache fallback."""
    kernel = _native.l2norm_bwd_kernel
    rows = y.shape[1] * 32
    arguments = dict(y=y, rstd=SimpleNamespace(dtype=torch.float32), dy=dy, dx=y)
    arguments.update(eps=1e-6, T=rows, D=128, BD=128, NB=triton.cdiv(rows, 2048 * 32))
    key = _cache.AutotuneKey.build(kernel.arg_names, kernel.keys, (), arguments)
    if len(kernel.configs) == 1:
        config = kernel.configs[0]
    elif kernel.should_check_fla_cache(key):
        return None
    else:
        config = kernel.cache.get(key.autotune_key)
    if config is None or config.pre_hook is not None or set(config.kwargs) != {"BT"}:
        return None
    if config.kwargs["BT"] not in (8, 16, 32, 64, 128):
        return None
    return config


def _repeat_backward(y, rstd, dy):
    fused = os.environ.get("VEOMNI_MUSA_QK_FUSED_BACKWARD", "0") == "1"
    config = native_backward_config(y, dy) if fused else None
    if config is not None:
        dx16 = torch.empty((1, y.shape[1], 16, 128), dtype=y.dtype, device=y.device)
        pairs = y.shape[1] * 16
        with torch.musa.device(y.device):
            qk_pair_l2norm_bwd_kernel[(triton.cdiv(pairs * 2, config.kwargs["BT"]),)](
                y, rstd, dy.contiguous(), dx16, pairs, D=128, **config.all_kwargs()
            )
        stats["fused_pair_backward"] += 1
        stats[f"BW_BT{config.kwargs['BT']}_W{config.num_warps}_S{config.num_stages}"] += 1
        stats["fused_backward"] += 1
        return dx16
    if fused:
        stats["native_backward_config_fallback"] += 1
    # The native input_guard materializes dy; preserve that before the helper.
    with torch.musa.device(y.device):
        dx32 = _native.l2norm_bwd(y, rstd, dy.contiguous(), 1e-6)
    assert dx32.dtype == torch.bfloat16
    dx16 = dx32.view(1, y.shape[1], 16, 2, 128).sum(dim=3)
    stats["fused_backward"] += 1
    return dx16


class _RepeatL2Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, config):
        shape = (1, x.shape[1], 32, 128)
        y = torch.empty(shape, dtype=x.dtype, device=x.device)
        rstd = torch.empty(shape[:-1], dtype=torch.float32, device=x.device)
        rows = x.shape[1] * 32
        with torch.musa.device(x.device):
            qk_repeat_l2norm_fwd_kernel[(triton.cdiv(rows, config.kwargs["BT"]),)](
                x=x,
                y=y,
                rstd=rstd,
                eps=1e-6,
                T=rows,
                D=128,
                BD=128,
                NB=triton.cdiv(rows, 2048 * 32),
                STRIDE_T=x.stride(1),
                **config.all_kwargs(),
            )
        ctx.save_for_backward(y, rstd)
        stats["fused_forward"] += 1
        stats[f"FW_BT{config.kwargs['BT']}_W{config.num_warps}_S{config.num_stages}"] += 1
        return y

    @staticmethod
    def backward(ctx, dy):
        if torch.is_grad_enabled():
            raise RuntimeError("MUSA repeat normalization does not support higher-order gradients")
        y, rstd = ctx.saved_tensors
        return _repeat_backward(y, rstd, dy), None


def repeat_norm(x, eps=1e-6):
    initialize()
    if not eligible(x, eps):
        stats["fallback"] += 1
        return native(x, eps)
    config = native_config(x)
    if config is None:
        stats["fallback"] += 1
        stats["native_config_fallback"] += 1
        return native(x, eps)
    return _RepeatL2Norm.apply(x, config)


class _JointRepeatL2Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, config):
        shape = (1, query.shape[1], 32, 128)
        yq, yk = (torch.empty(shape, dtype=query.dtype, device=query.device) for _ in range(2))
        rq, rk = (torch.empty(shape[:-1], dtype=torch.float32, device=query.device) for _ in range(2))
        rows = query.shape[1] * 32
        with torch.musa.device(query.device):
            qk_joint_l2norm_fwd_kernel[(triton.cdiv(rows, config.kwargs["BT"]), 2)](
                query, key, yq, yk, rq, rk, rows, D=128, STRIDE_T=query.stride(1), **config.all_kwargs()
            )
        ctx.save_for_backward(yq, rq, yk, rk)
        stats["joint_forward"] += 1
        stats["fused_forward"] += 2
        return yq, yk

    @staticmethod
    def backward(ctx, dyq, dyk):
        if torch.is_grad_enabled():
            raise RuntimeError("MUSA repeat normalization does not support higher-order gradients")
        yq, rq, yk, rk = ctx.saved_tensors
        return _repeat_backward(yq, rq, dyq), _repeat_backward(yk, rk, dyk), None


def repeat_norm_pair(query, key):
    initialize()
    if eligible(query, 1e-6) and eligible(key, 1e-6) and query.shape == key.shape and query.device == key.device:
        qc, kc = native_config(query), native_config(key)
        if qc is not None and kc is not None and qc.all_kwargs() == kc.all_kwargs():
            return _JointRepeatL2Norm.apply(query, key, qc)
    stats["joint_fallback"] += 1
    return repeat_norm(query), repeat_norm(key)
