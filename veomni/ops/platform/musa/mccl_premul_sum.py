"""MCCL compatibility for ReduceOp.PREMUL_SUM.

Some torch/FSDP2 releases emit this reduction operation while MCCL only accepts
SUM.  Rewriting to SUM and scaling the completed output is equivalent for the
reduction sites used by VeOmni and mirrors the existing HCCL shim.

``ReduceOp.AVG`` is passed through untouched by default, matching what the NPU shim
(:mod:`veomni.ops.platform.npu.hccl_premul_sum`) has always done.  MCCL implements it
natively on torch_musa 2.9.1, bit-for-bit identically to ``SUM`` plus a separate
``mul_(1 / group_size)``, but without the extra elementwise pass and the
``handle.wait()`` sync the rewrite adds per reduce-scatter.  Measured on
Qwen3.5-35B-A3B, 8x MTT S5000: ``mul_`` launches from the FSDP2 post-backward hook
drop from 107 to 40 per step, and the step from 3.610 to 3.560 s, with identical
per-epoch loss.

That is safe by construction on any build that can train at all: this patch is only
installed when extra parallelism is enabled, while FSDP2 emits ``ReduceOp.AVG``
whenever ``reduce_dtype`` is a float type regardless of EP — so a non-EP MUSA run
already reaches MCCL with AVG and no shim, and would fail loudly if MCCL lacked it.
The rewrite is still reachable as an escape hatch: set ``VEOMNI_MCCL_NATIVE_AVG=0``
to restore ``AVG -> SUM + mul_(1 / group_size)`` for a build whose MCCL rejects AVG.
"""

import os
from functools import wraps
from typing import Any, Callable, Optional, Tuple

import torch
from torch.distributed.distributed_c10d import ReduceOp


_PATCHED = False
_CUSTOM_OVERLAP_PATCHED = False


def _native_avg_enabled() -> bool:
    """Whether MCCL's own ``ReduceOp.AVG`` may be used (``VEOMNI_MCCL_NATIVE_AVG=0`` opts out)."""
    return os.environ.get("VEOMNI_MCCL_NATIVE_AVG", "1").strip().lower() not in {"0", "false", "no", "off"}


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


def _wrap_custom_overlap_reduce_scatter(collective: Callable) -> Callable:
    """Add PREMUL_SUM handling to torch_musa's low-contention FSDP path."""

    @wraps(collective)
    def wrapper(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: Any,
        op: Any,
        async_op: bool = False,
    ):
        factor = _premul_factor(op)
        if factor is None:
            return collective(
                self,
                output_tensor=output_tensor,
                input_tensor=input_tensor,
                group=group,
                op=op,
                async_op=async_op,
            )

        # low_contention_reduce_scatter accepts a string reduction mode and
        # only implements sum/avg. Run the sum on the original stream, then
        # scale the output in that same stream to preserve FSDP ordering.
        handle = collective(
            self,
            output_tensor=output_tensor,
            input_tensor=input_tensor,
            group=group,
            op=ReduceOp.SUM,
            async_op=async_op,
        )
        if handle is not None:
            handle.wait()
        with torch.no_grad():
            output_tensor.mul_(factor)
        return handle

    wrapper._mccl_premul_sum_compatible = True
    return wrapper


def _patch_custom_overlap_reduce_scatter() -> None:
    """Patch torch_musa's COMM_TYPE=1 reduce-scatter implementation."""
    global _CUSTOM_OVERLAP_PATCHED
    if _CUSTOM_OVERLAP_PATCHED:
        return

    try:
        from torch_musa.distributed._composable.fsdp import custom_overlap_patch
    except ImportError:
        # Older torch_musa builds may not ship the custom overlap module.
        _CUSTOM_OVERLAP_PATCHED = True
        return

    comm_cls = getattr(
        custom_overlap_patch,
        "IntraNodeLowContentionCommReduceScatter",
        None,
    )
    if comm_cls is None:
        _CUSTOM_OVERLAP_PATCHED = True
        return

    collective = comm_cls.__call__
    if not getattr(collective, "_mccl_premul_sum_compatible", False):
        comm_cls.__call__ = _wrap_custom_overlap_reduce_scatter(collective)
    _CUSTOM_OVERLAP_PATCHED = True


def mccl_reduce_op_wrapper(op: Callable, output_name: str, op_arg_index: int, group_arg_index: int):
    """Wrap a collective, translating unsupported reduction operators."""

    def wrapper(*args, **kwargs):
        reduce_op = _extract_op(args, kwargs, op_arg_index)
        # ReduceOp.AVG is passed through untouched unless the escape hatch asks otherwise:
        # MCCL implements it natively (see the module docstring).  PREMUL_SUM, which MCCL
        # rejects outright, always needs the rewrite.
        factor = _premul_factor(reduce_op)
        if factor is None and not _native_avg_enabled() and _is_avg(reduce_op):
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
    _patch_custom_overlap_reduce_scatter()
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
