# Copyright 2026 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""MUSA platform integration shims.

The MUSA runtime exposes a CUDA-like tensor API but uses MCCL for process
group collectives.  These helpers keep the hardware-specific compatibility
workarounds out of the model and trainer code.
"""

from .flash_attn import apply_musa_flash_attn_patch
from .mccl_premul_sum import apply_mccl_premul_sum_patch, mccl_reduce_op_wrapper


__all__ = [
    "apply_musa_flash_attn_patch",
    "apply_mccl_premul_sum_patch",
    "mccl_reduce_op_wrapper",
]
