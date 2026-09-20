# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MUSA-only runtime patches for Qwen3.5-MoE.

Like :mod:`..qwen3_5.qwen3_5_musa_runtime_patch`, this is a small runtime
installer rather than a patchgen ``PatchConfig``: the GPU/NPU generated files
stay untouched and the behaviour is added on the active MUSA model path only.
"""

from __future__ import annotations

import inspect
from functools import wraps
from types import ModuleType

import torch
import torch.nn.functional as F

from ....distributed.parallel_state import get_parallel_state
from ....utils.dist_utils import all_reduce
from ....utils.logging import get_logger


logger = get_logger(__name__)


# Set on the vision model for the duration of one outer forward call to the
# number of `dummy_forward` calls it still has to serve.
_BUDGET_ATTR = "_veomni_dummy_forward_budget"

# Set on the modeling module to the installed patch-embed variant ("conv3d" or "linear").
# The install is process-global, so this records what the first build in the process chose.
_PATCH_EMBED_FOLD_ATTR = "_VEOMNI_MUSA_VISION_PATCH_EMBED_FOLD"


def _required_dummy_passes(
    has_image: bool,
    has_video: bool,
    *,
    group=None,
) -> int:
    """How many ``dummy_forward`` calls this rank has to serve for the current batch.

    Under FSDP every rank must invoke the vision tower the same number of times, or the
    gradient reduce-scatters desynchronise. A modality slot contributes one pass on a rank
    iff that rank holds the modality *or some other rank does*, so the dummies a rank owes
    are exactly the slots it is missing that somebody else has.

    The one deliberate departure is the all-text batch (no rank holds any modality), where
    the formula owes nothing on any rank. Returning ``1`` there keeps the tower touched:
    the generated forward's unconditional dummy used to guarantee that, and multi-rank DDP
    (``fsdp_enabled`` is a *size* check, so it is on for ``dp_mode="ddp"`` too) fails on an
    unused trainable parameter. Every rank sees the same all-text batch, so one pass each
    stays uniform.

    Both flags travel in a single all-reduce, and the caller must not short-circuit it on
    the rank's own flags: the number of collectives a rank issues cannot depend on its
    batch, or the ranks deadlock instead of merely disagreeing.
    """
    any_image, any_video = all_reduce([int(bool(has_image)), int(bool(has_video))], op="max", group=group)
    if not any_image and not any_video:
        return 1
    return int(bool(any_image) and not has_image) + int(bool(any_video) and not has_video)


def install_qwen3_5_moe_dummy_forward_skip_patch(
    modeling_module: ModuleType,
    *,
    enabled: bool = True,
) -> None:
    """Let a single-modality batch skip the empty slot's dummy vision-tower pass.

    ``Qwen3_5MoeModel.forward`` walks one vision-tower slot per modality: the image slot
    runs a real forward, or ``dummy_forward`` when this rank holds no image, and the video
    slot does the same. The dummies exist so that every FSDP rank invokes the tower the
    same number of times — an uneven count desynchronises the reduce-scatters and hangs.

    On an image-only dataset the video dummy is pure overhead: a pass over every vision
    block, forward *and* backward, whose result is multiplied by zero before it reaches
    ``inputs_embeds``. It cannot simply be dropped from the forward body, because the
    *count* is what FSDP balances and a rank holding an image still needs the image slot
    while a text-only rank needs its image dummy.

    What the count must be is decided cluster-wide by :func:`_required_dummy_passes`. The
    dummy call sites are interchangeable for the balance and contribute zero either way, so
    this patch serves that budget instead of always serving both: the first N
    ``dummy_forward`` calls run the tower as before and the remainder return a zero
    stand-in ``pooler_output``.

    ``enabled=False`` (from ``model.ops_implementation.skip_empty_modality_dummy``) leaves
    the generated forward untouched. The install is process-global and one-shot: the first
    Qwen3.5-MoE model built decides for every later one.
    """
    if not enabled:
        logger.info_rank0(
            "skip_empty_modality_dummy=false: keeping the generated forward's unconditional "
            "dummy vision-tower pass; expect twice the vision-tower collectives per step on a "
            "single-modality dataset."
        )
        return

    if getattr(modeling_module, "_VEOMNI_MUSA_DUMMY_FORWARD_SKIP_PATCHED", False):
        return

    model_cls = getattr(modeling_module, "Qwen3_5MoeModel", None)
    vision_cls = getattr(modeling_module, "Qwen3_5MoeVisionModel", None)
    output_cls = getattr(modeling_module, "BaseModelOutputWithPooling", None)
    if model_cls is None or vision_cls is None or output_cls is None:
        raise RuntimeError("Qwen3.5-MoE generated module is missing Qwen3_5MoeModel/Qwen3_5MoeVisionModel")

    original_model_forward = model_cls.forward
    original_dummy_forward = vision_cls.dummy_forward
    forward_signature = inspect.signature(original_model_forward)

    def modality_presence(self, args, kwargs) -> tuple:
        """Read the batch's modalities the way the generated forward binds them.

        ``pixel_values`` / ``pixel_values_videos`` are ordinary parameters, so a positional
        caller would otherwise read as "no modality" — and under-serving a dummy hangs the
        job, it does not merely slow it down. Anything unparseable therefore counts as
        present, which reproduces the unpatched behaviour.
        """
        try:
            bound = forward_signature.bind_partial(self, *args, **kwargs).arguments
        except TypeError:
            return True, True
        return bound.get("pixel_values") is not None, bound.get("pixel_values_videos") is not None

    @wraps(original_model_forward)
    def model_forward(self, *args, **kwargs):
        parallel_state = get_parallel_state()
        if not parallel_state.fsdp_enabled:
            return original_model_forward(self, *args, **kwargs)

        budget = _required_dummy_passes(
            *modality_presence(self, args, kwargs),
            group=parallel_state.fsdp_group,
        )
        vision = self.visual
        previous_budget = getattr(vision, _BUDGET_ATTR, None)
        setattr(vision, _BUDGET_ATTR, budget)
        try:
            return original_model_forward(self, *args, **kwargs)
        finally:
            setattr(vision, _BUDGET_ATTR, previous_budget)

    @wraps(original_dummy_forward)
    def dummy_forward(self):
        budget = getattr(self, _BUDGET_ATTR, None)
        if budget is not None:
            if budget <= 0:
                # Same contribution as the real dummy — `pooler_output.mean() * 0.0` — without
                # paying for a tower pass this rank does not need. The caller reads only
                # `pooler_output`, so the remaining output fields are left unset.
                return output_cls(pooler_output=torch.zeros((), dtype=self.dtype, device=self.device))
            setattr(self, _BUDGET_ATTR, budget - 1)
        return original_dummy_forward(self)

    model_cls.forward = model_forward
    vision_cls.dummy_forward = dummy_forward
    modeling_module._VEOMNI_MUSA_DUMMY_FORWARD_SKIP_PATCHED = True


def install_qwen3_5_moe_vision_patch_embed_linear_patch(
    modeling_module: ModuleType,
    *,
    enabled: bool = True,
) -> None:
    """Rewrite the ViT patch embedder as one GEMM instead of a ``Conv3d``.

    ``Qwen3_5MoeVisionPatchEmbed`` is ``Conv3d(3 -> 1152, kernel=(2,16,16), stride=(2,16,16),
    padding=0)`` applied to ``hidden_states.view(N, 3, 2, 16, 16)``.  With ``kernel == stride``
    and no padding the output is spatially 1x1x1, so the convolution is *exactly* a matrix
    product over the flattened window:

        out[n, m] = sum_{c,t,h,w} w[m, c, t, h, w] * x[n, c, t, h, w] + b[m]
                  = F.linear(x.view(N, 1536), w.view(1152, 1536), b)

    The two spellings are the same arithmetic in a different reduction order, and on MUSA they
    are not remotely the same speed.  Measured on one MTT S5000 at the shapes this trainer uses
    (bf16, N=18432 patches, forward + weight gradient; ``pixel_values`` never requires grad, so
    there is no input gradient):

    ================================  ============  ============  ========
    kernel                            conv3d        linear        gain
    ================================  ============  ============  ========
    patch embed forward               1.678 ms      0.183 ms      9.2x
    forward + weight gradient         100.264 ms    0.465 ms      216x
    effective wgrad throughput        0.7 TFLOP/s   231 TFLOP/s   330x
    ================================  ============  ============  ========

    ``implicit_gemm_conv3d_bwd_filter_ndhwc_128_128x128`` launches a 128-thread block with
    64 KiB of shared memory, which caps occupancy at 13% and lands ~0.14% of the dense bf16
    peak.  The folded form instead reaches the ordinary tensor-core GEMM path, and also drops
    the NCDHW->NDHWC layout copies the conv path pays for on every call.

    Numerically the fold is *at least* as accurate as the convolution.  Against an fp64
    reference, ``linear`` sits at 7.0e-7 rms relative while MUSA's ``conv3d`` sits at 7.3e-4 --
    its implicit-gemm kernel accumulates in reduced precision, so the ~5e-5 rms_rel gap between
    the two spellings is the convolution's own accumulation error, not fp32 re-association.
    In bf16 the two agree to one ULP on the worst element (rms relative 5e-5 forward, 7e-5
    weight gradient), far below the shared bf16 input-quantization error of 1.7e-3.  The swap
    is therefore not bit-neutral and A/B loss curves will differ in the last digits, but the
    difference is bounded by the bf16 noise both paths already carry.

    The parameter keeps its ``nn.Conv3d`` layout, so checkpoints, ``state_dict`` keys and the
    ``parallel_plan`` are untouched; only the forward body changes.  FSDP2 is unaffected: the
    ``reshape`` is a plain merge of the replicated trailing dims, and the weight gradient comes
    back in the parameter's own ``(1152, 3, 2, 16, 16)`` shape.

    ``enabled=False`` (from ``model.ops_implementation.vision_patch_embed_implementation='conv3d'``)
    is the default and leaves the generated forward untouched.  The install is process-global:
    the first Qwen3.5-MoE model built decides for every later one, and a later build requesting
    the other value is warned about rather than silently ignored.
    """
    installed = getattr(modeling_module, _PATCH_EMBED_FOLD_ATTR, None)
    requested = "linear" if enabled else "conv3d"
    if installed is not None:
        if installed != requested:
            logger.warning_rank0(
                f"vision_patch_embed_implementation={requested} ignored: this process already "
                f"installed the Qwen3.5-MoE ViT patch embedder as '{installed}'. The choice is "
                f"process-global and one-shot; restart to change it."
            )
        return

    if not enabled:
        setattr(modeling_module, _PATCH_EMBED_FOLD_ATTR, requested)
        return

    patch_embed_cls = getattr(modeling_module, "Qwen3_5MoeVisionPatchEmbed", None)
    if patch_embed_cls is None:
        raise RuntimeError("Qwen3.5-MoE generated module is missing Qwen3_5MoeVisionPatchEmbed")

    original_forward = patch_embed_cls.forward

    @wraps(original_forward)
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = self.proj.weight
        # A (N, C, T, P, P) block and ``weight.reshape(embed_dim, -1)`` flatten in the same
        # (c, t, h, w) order, which is what makes the fold exact.  ``reshape`` rather than
        # ``view`` so a non-contiguous weight degrades to a copy instead of raising -- the
        # upstream ``conv3d`` accepts such a layout, and nothing should become rejected here.
        hidden_states = hidden_states.reshape(-1, self.in_channels * self.temporal_patch_size * self.patch_size**2)
        hidden_states = hidden_states.to(dtype=weight.dtype)
        return F.linear(hidden_states, weight.reshape(self.embed_dim, -1), self.proj.bias)

    patch_embed_cls.forward = forward
    setattr(modeling_module, _PATCH_EMBED_FOLD_ATTR, requested)
    logger.info_rank0(
        "vision_patch_embed_implementation=linear: Qwen3.5-MoE ViT patch embed folded into a single GEMM."
    )


__all__ = [
    "install_qwen3_5_moe_dummy_forward_skip_patch",
    "install_qwen3_5_moe_vision_patch_embed_linear_patch",
]
