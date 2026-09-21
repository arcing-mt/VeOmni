"""MUSA batched gradient-norm reduction for the FSDP2 gradient-clip path.

The shared implementation in ``veomni/distributed/fsdp2/clip_grad_norm.py``
accumulates the local gradient norm one parameter at a time -- one
``torch.linalg.vector_norm``, one ``pow`` and one ``add`` each, i.e. ~3078 serial
launches per step for a 35B MoE model.  On Qwen3.5-35B-A3B with ``ep_size=8``
that burst spans 119.5 ms of the step while the device is busy for only 49.9 ms
of it, and kernels launched by other work account for 0.08 ms of the span: the
~50 ms of device work is fully exposed and the remaining ~70 ms is launch
starvation.  Batching the same reduction into one ``_foreach_norm`` plus one
``stack -> pow -> sum`` measures 52.6-60.4 ms -> 7.0-8.4 ms on a synthetic
1026-parameter set, and 8x MTT S5000 end-to-end medians over 19 steps give
3.540/3.560 s -> 3.470/3.510 s (-50 to -70 ms per step) with an unchanged loss.

The change lives in the MUSA platform plugin instead of shared distributed code:

* :func:`apply_musa_fsdp2_clip_grad_norm_patch` rebinds ``_local_pth_sum`` and
  ``_local_max`` on the shared module.  Their only call site,
  ``_fsdp2_reduce_group``, resolves them by name at call time, so no shared file
  changes and no other backend is affected.
* Like the other MUSA platform patches it is installed unconditionally for MUSA
  runs from ``parallelize_model_fsdp2``, next to
  :func:`veomni.ops.platform.musa.mccl_premul_sum.apply_mccl_premul_sum_patch`.

Two details are easy to get wrong:

* ``torch._foreach_norm(..., dtype=torch.float32)`` accumulates in FP32 inside
  the reduce kernel.  ``torch.linalg.vector_norm(x, ord=p, dtype=torch.float32)``
  does *not*: it casts ``x`` first, so the per-parameter loop still allocates one
  FP32-sized temporary per gradient.  Since the loop frees each temporary before
  building the next, the peak growth is ``4 B x numel`` of the *largest single
  local gradient*, not of the gradients in total -- 242 MiB for this model's
  ``lm_head`` / ``embed_tokens`` shard (``248320 x 2048 / 8`` elements).  That is
  a peak-margin detail rather than a reason to take the patch: an 80 GB card that
  is already 44 GB full normally serves it from the caching allocator.
* Only ``ord`` in ``{1, 2, inf}`` have a batched kernel.  Torch forwards any
  other order (``norm_type=0``, ``3.5``, ...) to ``foreach_tensor_norm_slow``,
  which is one ``linalg_vector_norm`` per tensor again -- correct, just not
  batched.  The default ``norm_type=2.0`` takes the batched path.

Numerics are unchanged on MUSA: for ``ord`` in ``{0, 1, 1.5, 2, 3, inf}`` across
fp16, bf16 and fp32 the per-tensor norms agree with the per-parameter spelling to
within FP32 reduction order (worst observed 7.8e-8 relative, about one ulp; the
max path is bit-exact), and the summed p-th power differs by 1.5-1.9e-7 relative
-- below the 5e-7 seen over a 400-tensor mixed-dtype, mixed-device set that
includes missing and zero-element gradients.
"""

import importlib
import math
from typing import Any, List, Optional

import torch
from torch.distributed._tensor import DTensor

from ....utils.device import get_device_type


_PATCHED = False
_ORIGINALS: dict[str, Any] = {}

#: Dtypes whose reduction can be batched with an in-kernel FP32 accumulator.
#: ``torch._foreach_norm`` falls back to one ``linalg_vector_norm`` per tensor
#: unless every tensor of a call shares a device, a dtype and a dense layout.
_FOREACH_NORM_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _shared_clip_module():
    """Return the shared clip module itself.

    ``veomni.distributed.fsdp2.__init__`` re-exports the *function*
    ``clip_grad_norm``, which shadows the submodule of the same name, so the
    module has to be fetched through ``importlib`` rather than ``from ... import``.
    """
    return importlib.import_module("veomni.distributed.fsdp2.clip_grad_norm")


def _detached_local_grad(param: torch.nn.Parameter) -> Optional[torch.Tensor]:
    """Return ``param.grad`` as a plain, detached local tensor (``None`` if unset)."""
    g = param.grad
    if g is None:
        return None
    if isinstance(g, DTensor):
        g = g.to_local()
    return g.detach()


def _batched_norms(params: List[torch.nn.Parameter], ord: float, device: torch.device) -> List[torch.Tensor]:
    """Return one FP32 norm (of order ``ord``) per parameter that has a gradient.

    Every returned norm lives on ``device``, so the caller can stack them.
    """
    buckets: dict[tuple[str, torch.dtype], List[torch.Tensor]] = {}
    fallback: List[torch.Tensor] = []
    off_device = False
    for param in params:
        g = _detached_local_grad(param)
        if g is None:
            continue
        # Compare types: ``device`` is indexless (``musa``) while a live tensor
        # reports ``musa:0``, and the two never compare equal.
        off_device = off_device or g.device.type != device.type
        if g.numel() > 0 and g.dtype in _FOREACH_NORM_DTYPES:
            # ``_foreach_norm`` only takes its batched kernel when every tensor of
            # the list shares a device *and* a dtype; any other list is silently
            # reduced one tensor at a time, which is what this plugin exists to
            # remove.  Bucket explicitly, because a mixed-precision policy
            # (``modules_to_ignore_in_mixed_precision``) or an offloaded bucket is
            # enough to hand us such a list.
            buckets.setdefault((g.device.type, g.dtype), []).append(g)
        elif math.isinf(ord):
            # Mirror the shared spelling, which takes ``abs().max()`` rather than
            # ``vector_norm(ord=inf)``.  A zero-element gradient stays on this
            # spelling because ``_foreach_norm`` rejects one for ord=inf on CPU
            # and CUDA alike, while on MUSA ``abs().max()`` of an empty tensor is
            # 0.0 -- reproducing the shared behaviour on every backend.
            fallback.append(torch.abs(g.to(torch.float32)).max())
        else:
            # Wider or uncommon dtypes keep the explicit FP32 cast: the batched
            # reduction rejects e.g. float64 combined with ``dtype=torch.float32``
            # ("should be convertible without narrowing"), and integer gradients
            # only work after a cast.
            fallback.append(torch.linalg.vector_norm(g.to(torch.float32), ord=ord))
    norms: List[torch.Tensor] = []
    for bucket in buckets.values():
        norms.extend(torch._foreach_norm(bucket, ord=ord, dtype=torch.float32))
    norms.extend(fallback)
    if off_device:
        # ``torch.stack`` requires a single device, and the cpu-offload path can
        # mix gradients that were offloaded to the host with resident ones.
        # ``torch._foreach_norm`` itself has no such restriction.
        norms = [n.to(device=device, dtype=torch.float32) for n in norms]
    return norms


def musa_local_pth_sum(params: List[torch.nn.Parameter], p: float) -> torch.Tensor:
    """Batched replacement for ``clip_grad_norm._local_pth_sum``."""
    reduce_device = torch.device(get_device_type())
    with torch.no_grad():
        norms = _batched_norms(params, ord=p, device=reduce_device)
        if not norms:
            return torch.tensor(0.0, device=reduce_device, dtype=torch.float32)
        # One fused ``stack -> pow -> sum`` replaces the per-parameter
        # ``res = res + norm.pow(p)`` chain, which cost two extra launches and
        # one extra temporary per parameter.
        return torch.stack(norms).pow(p).sum()


def musa_local_max(params: List[torch.nn.Parameter]) -> torch.Tensor:
    """Batched replacement for ``clip_grad_norm._local_max``."""
    reduce_device = torch.device(get_device_type())
    with torch.no_grad():
        norms = _batched_norms(params, ord=float("inf"), device=reduce_device)
        if not norms:
            return torch.tensor(0.0, device=reduce_device, dtype=torch.float32)
        return torch.stack(norms).max()


def apply_musa_fsdp2_clip_grad_norm_patch() -> None:
    """Rebind the FSDP2 clip helpers to the batched MUSA reductions (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return

    clip_grad_norm_module = _shared_clip_module()

    _ORIGINALS["_local_pth_sum"] = clip_grad_norm_module._local_pth_sum
    _ORIGINALS["_local_max"] = clip_grad_norm_module._local_max
    clip_grad_norm_module._local_pth_sum = musa_local_pth_sum
    clip_grad_norm_module._local_max = musa_local_max
    _PATCHED = True


def revert_musa_fsdp2_clip_grad_norm_patch() -> None:
    """Restore the shared reductions installed over by :func:`apply_musa_fsdp2_clip_grad_norm_patch`."""
    global _PATCHED
    if not _PATCHED:
        return

    clip_grad_norm_module = _shared_clip_module()

    for name in ("_local_pth_sum", "_local_max"):
        original = _ORIGINALS.pop(name, None)
        if original is not None:
            setattr(clip_grad_norm_module, name, original)
    _PATCHED = False
