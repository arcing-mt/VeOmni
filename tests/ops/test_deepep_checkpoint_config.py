"""CLI/YAML contracts for selective DeepEP checkpoint caching."""

import argparse
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import yaml

from veomni.arguments.arguments_types import GradientCheckpointingConfig, OpsImplementationConfig, VeOmniArguments
from veomni.arguments.parser import _add_arguments_recursive, _deep_update, _instantiate_recursive
from veomni.distributed.moe import deepep_ace_checkpoint as cache


@dataclass
class _Model:
    ops_implementation: OpsImplementationConfig = field(default_factory=OpsImplementationConfig)


@dataclass
class _Root:
    model: _Model = field(default_factory=_Model)


def _parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    _add_arguments_recursive(parser, _Root)
    return parser


def test_cli_overrides_yaml_cache_policy_and_round_trips():
    data = yaml.safe_load("""
model:
  ops_implementation:
    moe_dispatcher: deepep_ace
    moe_deepep_recompute_cache: none
    moe_deepep_cache_layer_stride: 4
""")
    args = _parser().parse_args(
        [
            "--model.ops_implementation.moe_deepep_recompute_cache",
            "compact",
            "--model.ops_implementation.moe_deepep_cache_layer_stride",
            "2",
        ]
    )
    overrides = {key.removeprefix("model.ops_implementation."): value for key, value in vars(args).items()}
    _deep_update(data, {"model": {"ops_implementation": overrides}})
    config = _instantiate_recursive(_Root, data).model.ops_implementation
    assert config.moe_deepep_recompute_cache == "compact"
    assert config.moe_deepep_cache_layer_stride == 2
    from dataclasses import asdict

    restored = _instantiate_recursive(OpsImplementationConfig, yaml.safe_load(yaml.safe_dump(asdict(config))))
    assert restored.moe_deepep_recompute_cache == "compact"
    assert restored.moe_deepep_cache_layer_stride == 2


def test_cache_defaults_do_not_change_training():
    config = OpsImplementationConfig()
    assert config.moe_deepep_recompute_cache == "none"
    assert config.moe_deepep_cache_layer_stride == 2


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_invalid_stride_rejected_in_yaml(value):
    with pytest.raises(ValueError, match="positive integer"):
        _instantiate_recursive(OpsImplementationConfig, {"moe_deepep_cache_layer_stride": value})


def test_invalid_mode_rejected_by_cli_and_yaml():
    with pytest.raises(SystemExit):
        _parser().parse_args(["--model.ops_implementation.moe_deepep_recompute_cache", "layout"])
    with pytest.raises(ValueError, match="'none' or 'compact'"):
        _instantiate_recursive(OpsImplementationConfig, {"moe_deepep_recompute_cache": "layout"})


def test_compact_cache_requires_deepep():
    with pytest.raises(ValueError, match="moe_dispatcher='deepep_ace'"):
        OpsImplementationConfig(moe_deepep_recompute_cache="compact")


@pytest.mark.parametrize(
    "legacy",
    [
        "moe_deepep_reuse_layout_on_recompute",
        "moe_deepep_cache_dispatch_on_recompute",
        "moe_deepep_reuse_dispatch_handle_on_recompute",
    ],
)
def test_legacy_cli_and_yaml_are_rejected(legacy):
    with pytest.raises(SystemExit):
        _parser().parse_args([f"--model.ops_implementation.{legacy}", "true"])
    with pytest.raises(ValueError, match="was removed"):
        _instantiate_recursive(OpsImplementationConfig, {legacy: False})


def test_legacy_environment_cannot_enable_or_change_cache(monkeypatch):
    for name in [
        "VEOMNI_DEEPEP_CACHE_DISPATCH_ON_RECOMPUTE",
        "VEOMNI_DEEPEP_CACHE_COMPACT_DISPATCH",
        "VEOMNI_DEEPEP_REUSE_DISPATCH_HANDLE_ON_RECOMPUTE",
        "VEOMNI_DEEPEP_CACHE_LAYER_STRIDE",
        "VEOMNI_DEEPEP_REUSE_LAYOUT_VERIFY",
    ]:
        monkeypatch.setenv(name, "1")
    config = OpsImplementationConfig(moe_dispatcher="deepep_ace")
    monkeypatch.setattr(cache, "get_ops_config", lambda: config)
    assert not cache.recompute_cache_enabled()
    assert config.moe_deepep_cache_layer_stride == 2
    forward, recompute = cache.deepep_recompute_context_fn()
    with forward:
        assert not cache.has_cached_dispatch()
    with recompute:
        assert not cache.has_cached_dispatch()
    config.moe_deepep_recompute_cache = "compact"
    monkeypatch.setenv("VEOMNI_DEEPEP_CACHE_DISPATCH_ON_RECOMPUTE", "0")
    assert cache.recompute_cache_enabled()


@pytest.mark.parametrize(
    "checkpoint, compile_enabled",
    [
        (GradientCheckpointingConfig(enable=False), False),
        (GradientCheckpointingConfig(enable_reentrant=True), False),
        (GradientCheckpointingConfig(debug=True), False),
        (GradientCheckpointingConfig(), True),
    ],
)
def test_incompatible_checkpoint_configuration_rejected(checkpoint, compile_enabled):
    args = VeOmniArguments.__new__(VeOmniArguments)
    args.model = SimpleNamespace(ops_implementation=SimpleNamespace(moe_deepep_recompute_cache="compact"))
    args.train = SimpleNamespace(
        gradient_checkpointing=checkpoint, torch_compile=SimpleNamespace(enable=compile_enabled)
    )
    with pytest.raises(ValueError, match="DeepEP compact caching"):
        args.__post_init__()
