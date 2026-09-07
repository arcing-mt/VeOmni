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

import veomni.ops  # noqa: F401 - trigger kernel registrations
from veomni.ops.kernel_registry import KERNEL_REGISTRY


def test_musa_chunk_gated_delta_rule_is_registered() -> None:
    assert "musa" in KERNEL_REGISTRY.list_available("chunk_gated_delta_rule", "standard")


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
