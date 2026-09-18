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

from ....distributed.parallel_state import get_parallel_state
from ....utils.dist_utils import all_reduce
from ....utils.logging import get_logger


logger = get_logger(__name__)


# Set on the vision model for the duration of one outer forward call to the
# number of `dummy_forward` calls it still has to serve.
_BUDGET_ATTR = "_veomni_dummy_forward_budget"


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


__all__ = ["install_qwen3_5_moe_dummy_forward_skip_patch"]
