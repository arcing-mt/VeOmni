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

"""The MUSA patch that scopes the collator's host-side GDN offsets to a forward.

The offsets cannot reach the kernel as an argument (the generated
``GatedDeltaNet.forward`` builds its own call), so the decoder-layer forward is
wrapped and the adapter picks the value up from a context variable. What must hold:

* the batch key is consumed by the wrapper and never leaks into the other branches
  (a full-attention layer forwards ``**kwargs`` to ``self_attn``);
* the value is bound exactly for the duration of that forward, so the
  gradient-checkpoint recompute -- which re-enters the same class attribute on the
  autograd thread -- re-binds it instead of reading a stale one;
* a batch without the key behaves exactly as before;
* the installer is idempotent and covers both Qwen3.5 and Qwen3.5-MoE.

CPU-only: the generated module is imported but nothing runs on a device -- the GDN
layer is stubbed, and the one test that uses the real generated forward reaches
only through a stub ``self``.
"""

from types import ModuleType, SimpleNamespace

import pytest
import torch

from veomni.models.transformers.qwen3_5 import qwen3_5_musa_runtime_patch as patch
from veomni.ops.kernels.gated_delta_rule.musa_tilelang import _forward_cu_seqlens_cpu


OFFSETS = [0, 300, 500, 950]


def _decoder_layer_base():
    """A minimal stand-in for the generated decoder body, with its own call log."""
    seen = []

    class _DecoderLayer:
        def forward(self, hidden_states, **kwargs):
            # Stands in for the generated body: read what the GDN adapter would read.
            seen.append((_forward_cu_seqlens_cpu.get(), dict(kwargs)))
            return "out"

    _DecoderLayer.seen = seen
    return _DecoderLayer


def _fake_module(*class_names):
    """A modeling module holding minimal decoder layers, one per requested name."""
    module = ModuleType("fake_modeling")

    for name in class_names:
        setattr(module, name, type(name, (_decoder_layer_base(),), {}))
    return module


class TestWrapper:
    def test_scopes_the_offsets_to_the_forward(self):
        module = _fake_module("Qwen3_5MoeDecoderLayer")
        patch.install_qwen3_5_gdn_cpu_seqlens_patch(module)
        layer = module.Qwen3_5MoeDecoderLayer()

        layer.forward(torch.zeros(1), linear_attn_cu_seq_lens_q_cpu=OFFSETS)

        value, kwargs = module.Qwen3_5MoeDecoderLayer.seen[-1]
        assert value == tuple(OFFSETS)
        assert "linear_attn_cu_seq_lens_q_cpu" not in kwargs, "the key must not reach the model body"
        assert _forward_cu_seqlens_cpu.get() is None, "and must not outlive the forward"

    def test_batch_without_the_key_is_untouched(self):
        module = _fake_module("Qwen3_5MoeDecoderLayer")
        patch.install_qwen3_5_gdn_cpu_seqlens_patch(module)
        layer = module.Qwen3_5MoeDecoderLayer()

        layer.forward(torch.zeros(1), cu_seq_lens_q=torch.tensor(OFFSETS))

        value, kwargs = module.Qwen3_5MoeDecoderLayer.seen[-1]
        assert value is None
        assert "cu_seq_lens_q" in kwargs, "every other kwarg is passed through"

    def test_scoped_value_is_thread_local(self):
        """The recompute runs on the autograd thread; it must not see this thread's value."""
        module = _fake_module("Qwen3_5MoeDecoderLayer")
        patch.install_qwen3_5_gdn_cpu_seqlens_patch(module)
        layer = module.Qwen3_5MoeDecoderLayer()
        seen_elsewhere = []

        def elsewhere():
            seen_elsewhere.append(_forward_cu_seqlens_cpu.get())
            layer.forward(torch.zeros(1), linear_attn_cu_seq_lens_q_cpu=[0, 7])
            seen_elsewhere.append(_forward_cu_seqlens_cpu.get())

        with patch.forward_cu_seqlens_cpu(OFFSETS):
            import threading

            thread = threading.Thread(target=elsewhere)
            thread.start()
            thread.join()

        assert seen_elsewhere == [None, None]
        assert module.Qwen3_5MoeDecoderLayer.seen[-1][0] == (0, 7)


class TestInstaller:
    def test_covers_both_model_families(self):
        module = _fake_module("Qwen3_5MoeDecoderLayer", "Qwen3_5DecoderLayer")
        patch.install_qwen3_5_gdn_cpu_seqlens_patch(module)

        assert getattr(module.Qwen3_5MoeDecoderLayer, "_VEOMNI_GDN_CPU_SEQLENS_PATCHED", False)
        assert getattr(module.Qwen3_5DecoderLayer, "_VEOMNI_GDN_CPU_SEQLENS_PATCHED", False)

    def test_is_idempotent(self):
        module = _fake_module("Qwen3_5MoeDecoderLayer")
        patch.install_qwen3_5_gdn_cpu_seqlens_patch(module)
        once = module.Qwen3_5MoeDecoderLayer.forward
        patch.install_qwen3_5_gdn_cpu_seqlens_patch(module)

        assert module.Qwen3_5MoeDecoderLayer.forward is once, "a second install must not double-wrap"
        module.Qwen3_5MoeDecoderLayer().forward(torch.zeros(1), linear_attn_cu_seq_lens_q_cpu=OFFSETS)
        assert module.Qwen3_5MoeDecoderLayer.seen[-1][0] == tuple(OFFSETS)

    def test_ignores_a_module_without_the_class(self):
        module = _fake_module()
        patch.install_qwen3_5_gdn_cpu_seqlens_patch(module)  # must not raise


@pytest.fixture
def generated_moe_module():
    """The real generated module, with the class attribute restored afterwards."""
    from veomni.models.transformers.qwen3_5_moe.generated import patched_modeling_qwen3_5_moe_gpu as module

    original = module.Qwen3_5MoeDecoderLayer.forward
    original_flag = getattr(module.Qwen3_5MoeDecoderLayer, "_VEOMNI_GDN_CPU_SEQLENS_PATCHED", None)
    yield module
    module.Qwen3_5MoeDecoderLayer.forward = original
    if original_flag is None:
        del module.Qwen3_5MoeDecoderLayer._VEOMNI_GDN_CPU_SEQLENS_PATCHED
    else:
        module.Qwen3_5MoeDecoderLayer._VEOMNI_GDN_CPU_SEQLENS_PATCHED = original_flag


class TestGeneratedDecoderLayer:
    """The wrapper has to hold against the real generated forward, not a stand-in."""

    @staticmethod
    def _layer(kind, recorder):
        def norm(x):
            return x

        def mlp(hidden_states):
            return hidden_states

        def linear_attn(**kwargs):
            recorder.append(_forward_cu_seqlens_cpu.get())
            return kwargs["hidden_states"]

        def self_attn(hidden_states, **kwargs):
            recorder.append(kwargs.get("linear_attn_cu_seq_lens_q_cpu", "<absent>"))
            return hidden_states, None

        return SimpleNamespace(
            layer_type=kind,
            input_layernorm=norm,
            post_attention_layernorm=norm,
            mlp=mlp,
            linear_attn=linear_attn,
            self_attn=self_attn,
        )

    def test_linear_attention_layer_sees_the_offsets(self, generated_moe_module):
        patch.install_qwen3_5_gdn_cpu_seqlens_patch(generated_moe_module)
        seen = []
        layer = self._layer("linear_attention", seen)

        generated_moe_module.Qwen3_5MoeDecoderLayer.forward(
            layer,
            torch.zeros(1, 4, 8),
            position_embeddings=None,
            cu_seq_lens_q=torch.tensor(OFFSETS, dtype=torch.int32),
            linear_attn_cu_seq_lens_q_cpu=OFFSETS,
        )

        assert seen == [tuple(OFFSETS)]

    def test_full_attention_layer_never_receives_the_key(self, generated_moe_module):
        patch.install_qwen3_5_gdn_cpu_seqlens_patch(generated_moe_module)
        seen = []
        layer = self._layer("full_attention", seen)

        generated_moe_module.Qwen3_5MoeDecoderLayer.forward(
            layer,
            torch.zeros(1, 4, 8),
            position_embeddings=(None, None),
            cu_seq_lens_q=torch.tensor(OFFSETS, dtype=torch.int32),
            linear_attn_cu_seq_lens_q_cpu=OFFSETS,
        )

        assert seen == ["<absent>"]
