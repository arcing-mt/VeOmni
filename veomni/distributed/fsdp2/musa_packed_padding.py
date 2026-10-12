"""Source-guarded alignment in existing FSDP packing, scoped by comm objects."""

import ast
import hashlib
import importlib
import inspect
import textwrap
from collections import Counter
from pathlib import Path

import torch
from torch.distributed.fsdp._fully_shard import _fsdp_collectives as collectives
from torch.distributed.fsdp._fully_shard import _fsdp_param_group as param_group

from .musa_collective_padding import calls, shapes


SOURCE_SHA = "b7a5a3e09647961f3f5275168d8c76b619bc2906cfba67151e0951e44180d0de"
_installed = False


def _aligned_length(n, dtype, device, group, comm, allowed=True):
    if (
        not allowed
        or not getattr(comm, "_veomni_pack_aligned", False)
        or device.type != "musa"
        or group.size() != 8
        or dtype not in (torch.bfloat16, torch.float32)
        or n < 1_000_000
        or torch.compiler.is_compiling()
    ):
        return n
    multiple = 8 if dtype == torch.bfloat16 else 4
    return ((n + multiple - 1) // multiple) * multiple


def _aligned_reduce_size(total, dtype, device, group, comm, allowed):
    world = group.size()
    assert total % world == 0
    return _aligned_length(total // world, dtype, device, group, comm, allowed) * world


def _packed_all_gather_copy_in(inputs, output, sizes, padded, rank):
    logical = sum(sizes)
    if padded == logical:
        return torch.ops.fsdp.all_gather_copy_in(inputs, output, sizes, padded, rank)
    source = output.narrow(0, padded * rank, padded)
    torch._foreach_copy_(torch.split(source[:logical], sizes), inputs)
    source[logical:].zero_()
    # Keep logical metadata with this particular communication allocation.
    output._veomni_logical_row = logical
    calls["packed_all_gather"] += 1
    shapes[f"AG_PACK:{logical}->{padded}:{output.dtype}"] += 1
    return source, output


def _logical_rows(output, world):
    rows = output.view(world, -1)
    logical = getattr(output, "_veomni_logical_row", None)
    return rows if logical is None else rows[:, :logical]


def _packed_reduce_copy_in(grads, output, world):
    logical = sum(collectives._get_dim0_padded_size(grad.size(), world).numel() for grad in grads)
    if output.numel() == logical:
        return collectives.foreach_reduce_scatter_copy_in(grads, output, world)
    extra = output.numel() - logical
    assert extra > 0 and extra % world == 0 and extra // world < 8
    # A tiny final input lets the original chunk-cat write aligned rank rows
    # in one pass, preserving its dtype conversion and parameter packing order.
    tail = grads[0].new_zeros((extra,))
    collectives.foreach_reduce_scatter_copy_in([*grads, tail], output, world)
    calls["packed_reduce_scatter"] += 1
    shapes[f"RS_PACK:{logical // world}->{output.numel() // world}:{output.dtype}"] += 1


class _PackingTransform(ast.NodeTransformer):
    def __init__(self):
        self.changed = Counter()

    def visit_Assign(self, node):
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return self.generic_visit(node)
        target = node.targets[0].id
        if target == "all_gather_input_numel":
            self.changed["ag_size"] += 1
            node.value = ast.Call(
                func=ast.Name(id="_aligned_length", ctx=ast.Load()),
                args=[
                    node.value,
                    *[ast.Name(id=x, ctx=ast.Load()) for x in ("dtype", "device", "group", "all_gather_comm")],
                ],
                keywords=[],
            )
        elif target == "reduce_scatter_input_numel":
            self.changed["rs_size"] += 1
            allowed = ast.parse(
                "all_reduce_group is None and not copy_reduce_output_to_default_pool "
                "and partial_reduce_output is None and all_reduce_hook is None",
                mode="eval",
            ).body
            node.value = ast.Call(
                func=ast.Name(id="_aligned_reduce_size", ctx=ast.Load()),
                args=[
                    node.value,
                    *[
                        ast.Name(id=x, ctx=ast.Load())
                        for x in ("reduce_dtype", "device", "reduce_scatter_group", "reduce_scatter_comm")
                    ],
                    allowed,
                ],
                keywords=[],
            )
        elif ast.unparse(node) == "all_gather_output = all_gather_output.view(world_size, -1)":
            self.changed["ag_out"] += 1
            node.value = ast.parse("_logical_rows(all_gather_output, world_size)", mode="eval").body
        return self.generic_visit(node)

    def visit_Call(self, node):
        name = ast.unparse(node.func)
        if name == "torch.ops.fsdp.all_gather_copy_in":
            self.changed["ag_in"] += 1
            node.func = ast.Name(id="_packed_all_gather_copy_in", ctx=ast.Load())
        elif name == "foreach_reduce_scatter_copy_in":
            self.changed["rs_in"] += 1
            node.func = ast.Name(id="_packed_reduce_copy_in", ctx=ast.Load())
        return self.generic_visit(node)


def install():
    global _installed
    if _installed:
        return
    digest = hashlib.sha256(Path(collectives.__file__).read_bytes()).hexdigest()
    if digest != SOURCE_SHA:
        raise RuntimeError(f"Unsupported FSDP collective source for packed padding: {digest}")
    expected = {
        "foreach_all_gather": {"ag_size": 1, "ag_in": 1},
        "foreach_all_gather_copy_out": {"ag_out": 1},
        "foreach_reduce": {"rs_size": 1, "rs_in": 1},
    }
    replacements = {}
    # MUSA's no-overlap wait_for_unshard imports copy_out by value as well.
    musa_patch = importlib.import_module("torch_musa.distributed._composable.fsdp.patch")
    aliases = [collectives, param_group, musa_patch]
    bindings = {}
    for name, changes in expected.items():
        original = getattr(collectives, name)
        bindings[name] = []
        for module in aliases:
            bound = getattr(module, name, None)
            if bound is None:
                continue
            if bound is not original:
                raise RuntimeError(
                    f"FSDP {name} binding changed in {module.__name__} before packed-padding installation"
                )
            bindings[name].append(module)
        tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
        transform = _PackingTransform()
        tree = transform.visit(tree)
        assert transform.changed == changes, (name, transform.changed)
        namespace = dict(inspect.unwrap(original).__globals__)
        namespace.update(
            _aligned_length=_aligned_length,
            _aligned_reduce_size=_aligned_reduce_size,
            _packed_all_gather_copy_in=_packed_all_gather_copy_in,
            _logical_rows=_logical_rows,
            _packed_reduce_copy_in=_packed_reduce_copy_in,
        )
        exec(compile(ast.fix_missing_locations(tree), "<musa-packed-fsdp>", "exec"), namespace)
        replacements[name] = namespace[name]
    for name, replacement in replacements.items():
        for module in bindings[name]:
            setattr(module, name, replacement)
    _installed = True
