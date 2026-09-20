"""Tests for the Qwen3.5-MoE MUSA ViT patch-embed fold (``Conv3d`` -> one ``F.linear``).

The change under test is a speed rewrite, so the invariants are all "it is still the same
op": the patch embedder is ``Conv3d(kernel=(T,P,P), stride=(T,P,P), padding=0)``, which is
spatially 1x1x1 and therefore exactly a matrix product over the flattened window. The folded
forward must reproduce the convolution's values and weight gradient, keep the parameter's own
shape, and be a true no-op when the option is left at its ``conv3d`` default.

CPU-only: the fold is a pure reshape + GEMM, and the MUSA hardware gate is mocked.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from veomni.models.transformers.qwen3_5_moe import qwen3_5_moe_musa_runtime_patch as patch
from veomni.models.transformers.qwen3_5_moe.generated import patched_modeling_qwen3_5_moe_gpu as modeling_module


# Small stand-in for the 35B vision config (hidden_size=1152, patch=16, temporal=2): the fold
# and the view/grad plumbing are dimension-independent, and the real size makes the CPU
# reference convolution needlessly slow.
_IN_CHANNELS, _TEMPORAL, _PATCH, _EMBED = 3, 2, 4, 32
_WINDOW = _IN_CHANNELS * _TEMPORAL * _PATCH * _PATCH
_NUM_PATCHES = 12
_INSTALLED_FLAG = patch._PATCH_EMBED_FOLD_ATTR

_MISSING = object()


@pytest.fixture(autouse=True)
def _isolate_patch_state():
    """The installer is process-global and one-shot, so undo it after every test.

    Both the patched ``forward`` and the module-level "already installed" marker have to be
    restored, or the first test to run decides the behaviour of every later one.
    """
    cls = modeling_module.Qwen3_5MoeVisionPatchEmbed
    original_forward = cls.forward
    original_flag = getattr(modeling_module, _INSTALLED_FLAG, _MISSING)
    yield
    cls.forward = original_forward
    if original_flag is _MISSING:
        if hasattr(modeling_module, _INSTALLED_FLAG):
            delattr(modeling_module, _INSTALLED_FLAG)
    else:
        setattr(modeling_module, _INSTALLED_FLAG, original_flag)


def _config():
    return SimpleNamespace(
        patch_size=_PATCH,
        temporal_patch_size=_TEMPORAL,
        in_channels=_IN_CHANNELS,
        hidden_size=_EMBED,
    )


def _patched_module():
    """Install the fold on the real generated class and hand back a fresh instance."""
    patch.install_qwen3_5_moe_vision_patch_embed_linear_patch(modeling_module)
    return modeling_module.Qwen3_5MoeVisionPatchEmbed(_config())


def _upstream_forward(module, hidden_states):
    """The generated forward, re-expressed so it stays correct if the patch is installed."""
    weight = module.proj.weight
    window = hidden_states.view(-1, _IN_CHANNELS, _TEMPORAL, _PATCH, _PATCH)
    return module.proj(window.to(dtype=weight.dtype)).view(-1, module.embed_dim)


@pytest.fixture
def patched_module():
    """A patched module; ``_isolate_patch_state`` restores the class afterwards."""
    return _patched_module()


class TestArithmeticIdentity:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward_matches_upstream(self, patched_module, dtype):
        """The folded GEMM must reproduce the convolution, up to reduction-order rounding."""
        module = patched_module.to(dtype)
        torch.manual_seed(0)
        hidden_states = (torch.rand(_NUM_PATCHES * _WINDOW) * 2 - 1).to(dtype)

        folded = module(hidden_states)
        reference = _upstream_forward(module, hidden_states)

        assert folded.shape == reference.shape == (_NUM_PATCHES, _EMBED)
        # bf16 rounds inputs to ~2^-9; the fp32 accumulation-order term is orders below that.
        atol = 2e-2 if dtype is torch.bfloat16 else 1e-5
        torch.testing.assert_close(folded, reference, atol=atol, rtol=atol)

    def test_weight_gradient_matches_upstream(self, patched_module):
        """Only the weight needs grad (pixel_values is a leaf), so that is the grad to pin."""
        module = patched_module
        torch.manual_seed(0)
        hidden_states = torch.rand(_NUM_PATCHES * _WINDOW) * 2 - 1
        grad_output = torch.randn(_NUM_PATCHES, _EMBED) * 0.02

        folded_grad = torch.autograd.grad(module(hidden_states), module.proj.weight, grad_output)[0]
        reference_grad = torch.autograd.grad(
            _upstream_forward(module, hidden_states), module.proj.weight, grad_output
        )[0]

        assert folded_grad.shape == module.proj.weight.shape
        torch.testing.assert_close(folded_grad, reference_grad, atol=2e-2, rtol=2e-2)

    def test_forward_returns_flat_patch_major_output(self, patched_module):
        """The vision model consumes (num_patches, embed_dim); the fold must keep that contract."""
        module = patched_module
        hidden_states = torch.rand(_NUM_PATCHES * _WINDOW)
        assert module(hidden_states).shape == (_NUM_PATCHES, _EMBED)

    def test_upstream_dtype_cast_is_preserved(self, patched_module):
        """``hidden_states.to(dtype=weight.dtype)`` must survive the rewrite.

        ``pixel_values`` reaches the vision tower as fp32; the fold's GEMM needs it in the
        weight's dtype, and a missing cast would silently upcast the GEMM instead.
        """
        module = patched_module.to(torch.bfloat16)
        hidden_states = torch.rand(_NUM_PATCHES * _WINDOW)  # fp32, as the collator provides it

        folded = module(hidden_states)
        reference = _upstream_forward(module, hidden_states)

        assert folded.dtype is torch.bfloat16
        torch.testing.assert_close(folded, reference, atol=2e-2, rtol=2e-2)

    def test_parameter_layout_is_untouched(self, patched_module):
        """Checkpoints and ``parallel_plan`` key off the Conv3d layout -- keep it."""
        module = patched_module
        assert isinstance(module.proj, torch.nn.Conv3d)
        assert module.proj.weight.shape == (_EMBED, _IN_CHANNELS, _TEMPORAL, _PATCH, _PATCH)


class TestInstall:
    def test_disabled_is_a_no_op(self):
        """The default must leave the generated forward object untouched."""
        original_forward = modeling_module.Qwen3_5MoeVisionPatchEmbed.forward
        patch.install_qwen3_5_moe_vision_patch_embed_linear_patch(modeling_module, enabled=False)
        assert modeling_module.Qwen3_5MoeVisionPatchEmbed.forward is original_forward

    def test_install_is_idempotent(self):
        """A second install must not wrap the first one again."""
        original_forward = modeling_module.Qwen3_5MoeVisionPatchEmbed.forward
        patch.install_qwen3_5_moe_vision_patch_embed_linear_patch(modeling_module)
        installed = modeling_module.Qwen3_5MoeVisionPatchEmbed.forward
        patch.install_qwen3_5_moe_vision_patch_embed_linear_patch(modeling_module)
        assert modeling_module.Qwen3_5MoeVisionPatchEmbed.forward is installed
        assert installed is not original_forward

    def test_missing_class_raises(self):
        """A generated module without the class must fail loudly, not silently no-op."""
        with pytest.raises(RuntimeError, match="Qwen3_5MoeVisionPatchEmbed"):
            patch.install_qwen3_5_moe_vision_patch_embed_linear_patch(SimpleNamespace())

    def test_conv3d_then_linear_cannot_be_reverted_in_process(self):
        """The choice is process-global: a later build asking for the other value is warned.

        Silently honouring the second request is impossible (the first build already decided)
        and silently *ignoring* it while logging the opposite is worse, so the mismatch has to
        be visible.
        """
        original_forward = modeling_module.Qwen3_5MoeVisionPatchEmbed.forward
        patch.install_qwen3_5_moe_vision_patch_embed_linear_patch(modeling_module, enabled=True)
        folded = modeling_module.Qwen3_5MoeVisionPatchEmbed.forward

        with mock.patch.object(patch.logger, "warning_rank0") as warning:
            patch.install_qwen3_5_moe_vision_patch_embed_linear_patch(modeling_module, enabled=False)

        assert modeling_module.Qwen3_5MoeVisionPatchEmbed.forward is folded
        assert modeling_module.Qwen3_5MoeVisionPatchEmbed.forward is not original_forward
        assert "already installed" in warning.call_args.args[0]

    def test_repeated_install_with_the_same_value_is_silent(self):
        """The common path must not warn: the same build value installed twice is a no-op."""
        patch.install_qwen3_5_moe_vision_patch_embed_linear_patch(modeling_module, enabled=True)
        with mock.patch.object(patch.logger, "warning_rank0") as warning:
            patch.install_qwen3_5_moe_vision_patch_embed_linear_patch(modeling_module, enabled=True)
        warning.assert_not_called()

    def test_default_conv3d_does_not_warn(self):
        """``conv3d`` is the default, so recording it must not nag on every build."""
        with mock.patch.object(patch.logger, "warning_rank0") as warning:
            patch.install_qwen3_5_moe_vision_patch_embed_linear_patch(modeling_module, enabled=False)
        warning.assert_not_called()
        assert modeling_module._VEOMNI_MUSA_VISION_PATCH_EMBED_FOLD == "conv3d"


class TestConfigGate:
    """``vision_patch_embed_implementation`` is MUSA-only, so both gates live in the config."""

    @staticmethod
    def _config(**overrides):
        from veomni.arguments.arguments_types import OpsImplementationConfig

        # ``eager`` keeps the Triton package check out of an unrelated assertion.
        return OpsImplementationConfig(load_balancing_loss_implementation="eager", **overrides)

    def test_default_keeps_the_upstream_conv3d(self):
        assert self._config().vision_patch_embed_implementation == "conv3d"

    def test_linear_is_accepted_on_musa(self):
        with mock.patch("veomni.utils.import_utils.is_torch_musa_available", return_value=True):
            assert self._config(vision_patch_embed_implementation="linear").vision_patch_embed_implementation == (
                "linear"
            )

    def test_linear_is_rejected_off_musa(self):
        with mock.patch("veomni.utils.import_utils.is_torch_musa_available", return_value=False):
            with pytest.raises(ValueError, match="requires an active torch-musa/MUSA device"):
                self._config(vision_patch_embed_implementation="linear")

    @pytest.mark.parametrize("value", ["conv_3d", "gemm", "", "LINEAR"])
    def test_typos_are_rejected(self, value):
        with mock.patch("veomni.utils.import_utils.is_torch_musa_available", return_value=True):
            with pytest.raises(ValueError, match="is not supported"):
                self._config(vision_patch_embed_implementation=value)


class TestModelRegistrationWiring:
    """The field is worthless if the registration does not read it -- pin the polarity."""

    @staticmethod
    def _register(monkeypatch, implementation):
        from veomni.arguments.arguments_types import OpsImplementationConfig
        from veomni.models.transformers.qwen3_5_moe import register_qwen3_5_moe_modeling
        from veomni.ops.config import singleton as ops_singleton

        monkeypatch.setattr(ops_singleton, "_ops_config", None)
        monkeypatch.setattr(
            ops_singleton,
            "_ops_config",
            OpsImplementationConfig(
                load_balancing_loss_implementation="eager",
                # Leaves the sibling MUSA installer inert, so this test touches only the fold.
                skip_empty_modality_dummy=False,
                vision_patch_embed_implementation=implementation,
            ),
        )
        with mock.patch(
            "veomni.models.transformers.qwen3_5_moe.IS_MUSA_AVAILABLE",
            True,
        ):
            register_qwen3_5_moe_modeling("Qwen3_5MoeForConditionalGeneration")

    def test_linear_installs_the_fold(self, monkeypatch):
        original_forward = modeling_module.Qwen3_5MoeVisionPatchEmbed.forward
        self._register(monkeypatch, "linear")
        assert modeling_module._VEOMNI_MUSA_VISION_PATCH_EMBED_FOLD == "linear"
        assert modeling_module.Qwen3_5MoeVisionPatchEmbed.forward is not original_forward

    def test_conv3d_keeps_the_generated_forward(self, monkeypatch):
        original_forward = modeling_module.Qwen3_5MoeVisionPatchEmbed.forward
        self._register(monkeypatch, "conv3d")
        assert modeling_module._VEOMNI_MUSA_VISION_PATCH_EMBED_FOLD == "conv3d"
        assert modeling_module.Qwen3_5MoeVisionPatchEmbed.forward is original_forward
