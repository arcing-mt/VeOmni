"""MCCL compatibility for ReduceOp.PREMUL_SUM and ReduceOp.AVG.

Some torch/FSDP2 releases emit these reduction operations while MCCL only
accepts SUM.  Rewriting to SUM and scaling the completed output is equivalent
for the reduction sites used by VeOmni and mirrors the existing HCCL shim.
"""

from typing import Any, Callable, Optional, Tuple

import torch
from torch.distributed.distributed_c10d import ReduceOp


_PATCHED = False


def _state_value(state: Any) -> Any:
    return getattr(state, "value", state)


def _premul_factor(reduce_op: Any) -> Optional[float]:
    if reduce_op is None or not hasattr(reduce_op, "__getstate__"):
        return None
    state = reduce_op.__getstate__()
    premul_state = ReduceOp.PREMUL_SUM.__getstate__()
    if isinstance(state, tuple) and len(state) == 2:
        kind, factor = state
        if _state_value(kind) == premul_state:
            return float(factor)
    elif _state_value(state) == premul_state:
        return 1.0
    return None


def _is_avg(reduce_op: Any) -> bool:
    return (
        reduce_op is not None
        and hasattr(reduce_op, "__getstate__")
        and _state_value(reduce_op.__getstate__()) == ReduceOp.AVG.__getstate__()
    )


def _group_size(args: Tuple[Any, ...], kwargs: dict, group_arg_index: int) -> int:
    group = kwargs.get("group")
    if group is None and len(args) > group_arg_index:
        group = args[group_arg_index]
    if group is not None and hasattr(group, "size"):
        return group.size()
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size(group=group)


def _replace_op(args: Tuple[Any, ...], kwargs: dict, op_arg_index: int, reduce_op: Any):
    if "op" in kwargs:
        kwargs["op"] = reduce_op
        return args, kwargs
    if len(args) > op_arg_index:
        values = list(args)
        values[op_arg_index] = reduce_op
        return tuple(values), kwargs
    kwargs["op"] = reduce_op
    return args, kwargs


def _extract_op(args: Tuple[Any, ...], kwargs: dict, op_arg_index: int):
    if "op" in kwargs:
        return kwargs["op"]
    if len(args) > op_arg_index:
        return args[op_arg_index]
    return None


def mccl_reduce_op_wrapper(op: Callable, output_name: str, op_arg_index: int, group_arg_index: int):
    """Wrap a collective, translating unsupported reduction operators."""

    def wrapper(*args, **kwargs):
        reduce_op = _extract_op(args, kwargs, op_arg_index)
        factor = _premul_factor(reduce_op)
        if factor is None and _is_avg(reduce_op):
            factor = 1.0 / _group_size(args, kwargs, group_arg_index)
        if factor is not None:
            args, kwargs = _replace_op(args, kwargs, op_arg_index, ReduceOp.SUM)

        handle = op(*args, **kwargs)
        if factor is not None and handle is not None:
            handle.wait()
        if factor is not None:
            output = args[0] if args else kwargs[output_name]
            with torch.no_grad():
                output.mul_(factor)
        return handle

    return wrapper


def apply_mccl_premul_sum_patch() -> None:
    """Install the idempotent MCCL reduction wrappers."""
    global _PATCHED
    if _PATCHED:
        return
    torch.distributed.all_reduce = mccl_reduce_op_wrapper(
        torch.distributed.all_reduce, "tensor", op_arg_index=1, group_arg_index=2
    )
    torch.distributed.reduce_scatter = mccl_reduce_op_wrapper(
        torch.distributed.reduce_scatter, "output", op_arg_index=2, group_arg_index=3
    )
    torch.distributed.reduce_scatter_tensor = mccl_reduce_op_wrapper(
        torch.distributed.reduce_scatter_tensor, "output", op_arg_index=2, group_arg_index=3
    )
    _PATCHED = True
