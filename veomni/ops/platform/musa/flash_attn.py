"""Make Transformers' FA3 availability check understand MUSA.

The local MUSA FA3 wheel exposes the standard ``flash_attn_interface`` API,
but Transformers' generic predicate currently requires ``torch.cuda``.  The
patch only changes availability/diagnostic gates; actual attention execution
still goes through Transformers' FA3 wrapper and the installed MUSA wheel.
"""

import importlib.util


_PATCHED = False


def apply_musa_flash_attn_patch() -> bool:
    global _PATCHED
    if _PATCHED:
        return True

    import torch

    if not (hasattr(torch, "musa") and torch.musa.is_available()):
        return False
    if importlib.util.find_spec("flash_attn_interface") is None:
        return False

    from transformers import utils as transformers_utils
    from transformers.utils import import_utils

    def musa_aware_fa3_available() -> bool:
        return importlib.util.find_spec("flash_attn_interface") is not None and (
            torch.cuda.is_available() or (hasattr(torch, "musa") and torch.musa.is_available())
        )

    # Patch all module globals that Transformers captured with ``from ...
    # import``. The lru-cached original is deliberately bypassed.
    import_utils.is_flash_attn_3_available = musa_aware_fa3_available
    transformers_utils.is_flash_attn_3_available = musa_aware_fa3_available

    import transformers.modeling_flash_attention_utils as flash_utils

    flash_utils.is_flash_attn_3_available = musa_aware_fa3_available

    import transformers.modeling_utils as modeling_utils

    matrix = getattr(modeling_utils, "FLASH_ATTENTION_COMPATIBILITY_MATRIX", None)
    if matrix is not None and 3 in matrix:
        matrix[3]["general_availability_check"] = musa_aware_fa3_available
        supported = list(matrix[3].get("supported_devices", ()))
        if not any(name == "musa" for _, name in supported):
            supported.append((lambda: torch.musa.is_available(), "musa"))
            matrix[3]["supported_devices"] = tuple(supported)

    _PATCHED = True
    return True


__all__ = ["apply_musa_flash_attn_patch"]
