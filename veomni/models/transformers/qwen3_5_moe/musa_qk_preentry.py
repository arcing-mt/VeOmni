"""Source-guarded adapter for the qualified MUSA GDN preentry kernel."""

import ast
import functools
import hashlib
import importlib
import inspect
import os
import textwrap
from collections import Counter
from pathlib import Path

import torch

from . import qk_repeat_l2norm as prototype


MODULE = "veomni.models.transformers.qwen3_5_moe.generated.patched_modeling_qwen3_5_moe_gpu"
MODULE_SHA = "6983d604c7cbd5d10b310125e5dbbd6d5966ff2e81bf97e7b5a356d277730553"
FORWARD_SHA = "e9f0405ea191294dd4f622250bbb61a5b43df434cc74e890fb7958627ece36cb"
BACKEND_SHA = "558f09a880407a8b2cc283b271d656915f5dedc5685cfe62d7f4d6b57353f00c"
counts = Counter()
_original = None
_wrapper = None
_backend = None
_model_module = None


def supported(self, hidden_states, cache_params, cu_seq_lens_q):
    return (
        not getattr(self, "_qk_preentry_native_only", False)
        and self.chunk_gated_delta_rule is _backend
        and cache_params is None
        and not _model_module.get_parallel_state().ulysses_enabled
        and (self.num_k_heads, self.num_v_heads, self.head_k_dim, self.head_v_dim) == (16, 32, 128, 128)
        and type(hidden_states) is torch.Tensor
        and hidden_states.device.type == "musa"
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.ndim == 3
        and hidden_states.shape[0] == 1
        and hidden_states.shape[1] > 1
        and type(cu_seq_lens_q) is torch.Tensor
        and cu_seq_lens_q.dtype == torch.int32
        and cu_seq_lens_q.device == hidden_states.device
        and torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
    )


class _Transform(ast.NodeTransformer):
    def __init__(self):
        self.changed = Counter()

    def visit_Assign(self, node):
        if ast.unparse(node) == "query = query.contiguous()":
            self.changed["query_contiguous"] += 1
            return ast.parse("""
_qk_fast = _qk_eligible(query, 1e-6) and _qk_eligible(key, 1e-6)
if _qk_fast:
    query = _qk_repeat_norm(query)
    key = _qk_repeat_norm(key)
    _qk_counts['preentry_fast'] += 1
else:
    query = query.contiguous()
    key = key.contiguous()
    _qk_counts['preentry_native_layout'] += 1
""").body
        if ast.unparse(node) == "key = key.contiguous()":
            self.changed["key_contiguous"] += 1
            return None
        return self.generic_visit(node)

    def visit_If(self, node):
        if ast.unparse(node.test) == "self.num_v_heads // self.num_k_heads > 1":
            self.changed["repeat_guard"] += 1
            node.test = ast.BoolOp(
                op=ast.And(),
                values=[node.test, ast.UnaryOp(op=ast.Not(), operand=ast.Name(id="_qk_fast", ctx=ast.Load()))],
            )
        return self.generic_visit(node)

    def visit_Call(self, node):
        if ast.unparse(node.func) == "self.chunk_gated_delta_rule":
            for kw in node.keywords:
                if kw.arg == "use_qk_l2norm_in_kernel":
                    assert isinstance(kw.value, ast.Constant) and kw.value.value is True
                    kw.value = ast.UnaryOp(op=ast.Not(), operand=ast.Name(id="_qk_fast", ctx=ast.Load()))
                    self.changed["chunk_norm_flag"] += 1
        return self.generic_visit(node)


def install(modeling_module):
    global _original, _wrapper, _backend, _model_module
    module = modeling_module
    cls = module.Qwen3_5MoeGatedDeltaNet
    if _wrapper is not None:
        if cls.forward is not _wrapper:
            raise RuntimeError("GDN forward changed after q/k experiment installation")
        return
    backend_module = importlib.import_module("veomni.ops.kernels.gated_delta_rule.musa_tilelang")
    for target, expected in ((module, MODULE_SHA), (backend_module, BACKEND_SHA)):
        actual = hashlib.sha256(Path(target.__file__).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Unsupported q/k preentry source: {actual}")
    original = cls.forward
    source = textwrap.dedent(inspect.getsource(original))
    if hashlib.sha256(source.encode()).hexdigest() != FORWARD_SHA:
        raise RuntimeError("Actual loaded GDN forward does not match the reviewed source")
    tree = ast.parse(source)
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
    assert [ast.unparse(v) for v in tree.body[0].decorator_list] == ["force_accelerate_hooks('conv1d')"]
    transform = _Transform()
    tree = transform.visit(tree)
    assert transform.changed == {
        "query_contiguous": 1,
        "key_contiguous": 1,
        "repeat_guard": 1,
        "chunk_norm_flag": 1,
    }, transform.changed
    ast.fix_missing_locations(tree)
    namespace = dict(inspect.unwrap(original).__globals__)
    namespace.update(_qk_eligible=prototype.eligible, _qk_repeat_norm=prototype.repeat_norm, _qk_counts=counts)
    exec(compile(tree, "<qk-preentry-private-forward>", "exec"), namespace)
    private = namespace["forward"]
    prototype.initialize()
    _backend = backend_module.chunk_gated_delta_rule
    _model_module = module

    @functools.wraps(original)
    def wrapped(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None, cu_seq_lens_q=None):
        if os.environ.get("VEOMNI_MUSA_QK_PREENTRY", "0") == "1" and supported(
            self, hidden_states, cache_params, cu_seq_lens_q
        ):
            counts["eligible_forward"] += 1
            return private(self, hidden_states, cache_params, cache_position, attention_mask, cu_seq_lens_q)
        counts["original_forward"] += 1
        return original(self, hidden_states, cache_params, cache_position, attention_mask, cu_seq_lens_q)

    _original, _wrapper = original, wrapped
    cls.forward = wrapped
    counts["installed"] += 1


def original_forward(self, *args, **kwargs):
    return _original(self, *args, **kwargs)


def stats():
    return {"adapter": dict(counts), "prototype": dict(prototype.stats)}
