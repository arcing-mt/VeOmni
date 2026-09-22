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

import sys
from types import ModuleType

import pytest

import veomni.ops  # noqa: F401 - trigger kernel registrations
from veomni.ops.kernel_registry import KERNEL_REGISTRY


def test_musa_chunk_gated_delta_rule_is_registered() -> None:
    assert "musa" in KERNEL_REGISTRY.list_available("chunk_gated_delta_rule", "standard")


def test_musa_tilelang_chunk_gated_delta_rule_is_registered() -> None:
    from veomni.ops.kernel_registry import KERNEL_REGISTRY as registry

    assert "musa_tilelang" in registry.list_available("chunk_gated_delta_rule", "standard")


def test_musa_tilelang_backend_is_gated_on_musa_and_dependency() -> None:
    """Resolve succeeds only when both MUSA and torch_kernels are available.

    Asserts the enforced behaviour through ``resolve()`` rather than reading the
    ``device_type`` field back -- a field check would still pass if the gate itself
    stopped working.
    """
    import torch

    from veomni.ops.kernel_registry import KERNEL_REGISTRY as registry

    if hasattr(torch, "musa") and torch.musa.is_available():
        try:
            from torch_kernels.attention import gated_delta_net  # noqa: F401
        except ImportError:
            with pytest.raises(ImportError, match="torch_kernels"):
                registry.resolve("chunk_gated_delta_rule", "standard", "musa_tilelang")
        else:
            assert callable(registry.resolve("chunk_gated_delta_rule", "standard", "musa_tilelang"))
    else:
        with pytest.raises(RuntimeError, match="musa"):
            registry.resolve("chunk_gated_delta_rule", "standard", "musa_tilelang")


def test_musa_tilelang_adapter_disables_the_slow_torch_normalizer(monkeypatch) -> None:
    """The whole point of the adapter: it must NOT let torch_kernels normalize.

    ``torch_kernels`` implements ``use_qk_l2norm_in_kernel=True`` as a torch-level
    chain that costs ~4x FLA's Triton ``l2norm``; that is enough to turn the swap
    into a net loss, so the adapter normalizes itself and passes False.
    """
    import torch

    import veomni.ops.kernels.gated_delta_rule.musa_tilelang as adapter

    seen: dict = {}

    def fake_gated_delta_net(q, k, v, g, beta, **kwargs):
        seen.update(kwargs)
        return "out", None

    monkeypatch.setattr(adapter, "_require_torch_kernels", lambda: fake_gated_delta_net)
    monkeypatch.setattr("fla.modules.l2norm.l2norm", lambda x, **kw: x)
    sentinel = torch.zeros(1, requires_grad=True)
    # cu_seqlens is mandatory: the backend is packed-varlen training only.
    out, state = adapter.chunk_gated_delta_rule(
        sentinel,
        sentinel,
        sentinel,
        sentinel,
        sentinel,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=torch.tensor([0, 64, 128], dtype=torch.int32),
    )

    assert (out, state) == ("out", None)
    assert seen["use_qk_l2norm_in_kernel"] is False
    assert seen["backend"] == "tilelang"


def test_musa_adapter_returns_the_fla_callable_without_external_script(monkeypatch) -> None:
    import veomni.ops.kernels.gated_delta_rule.musa as musa

    fake_fla = ModuleType("fla")
    fake_fla_ops = ModuleType("fla.ops")
    fake_fla_gdr = ModuleType("fla.ops.gated_delta_rule")
    sentinel = object()
    fake_fla_gdr.chunk_gated_delta_rule = sentinel
    monkeypatch.setitem(sys.modules, "fla", fake_fla)
    monkeypatch.setitem(sys.modules, "fla.ops", fake_fla_ops)
    monkeypatch.setitem(sys.modules, "fla.ops.gated_delta_rule", fake_fla_gdr)
    monkeypatch.setattr(musa, "_ensure_tuned_fla", lambda: None)

    assert musa.get_musa_chunk_gated_delta_rule() is sentinel
