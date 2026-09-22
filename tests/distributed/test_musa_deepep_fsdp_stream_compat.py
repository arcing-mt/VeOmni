import sys
from types import ModuleType, SimpleNamespace

import torch

from veomni.distributed import torch_parallelize


def _install_fake_torch_musa(monkeypatch, custom_overlap_patch):
    package_names = [
        "torch_musa",
        "torch_musa.distributed",
        "torch_musa.distributed._composable",
        "torch_musa.distributed._composable.fsdp",
    ]
    packages = {}
    for name in package_names:
        package = ModuleType(name)
        package.__path__ = []
        packages[name] = package
        monkeypatch.setitem(sys.modules, name, package)
    packages[package_names[-1]].custom_overlap_patch = custom_overlap_patch


def test_musa_deepep_fsdp_stream_compat_keeps_copyin_on_current_stream(monkeypatch):
    from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPCommContext

    current_stream = object()
    created_streams = []

    def original_lazy_init(context, _device):
        context.all_gather_copy_in_stream = object()
        context.all_gather_stream = object()
        context.reduce_scatter_stream = object()

    custom_overlap_patch = SimpleNamespace(
        comm_context_lazy_init=original_lazy_init,
        _FSDP2_OVERLAP_LEVEL=SimpleNamespace(value=2),
    )
    _install_fake_torch_musa(monkeypatch, custom_overlap_patch)
    monkeypatch.setattr(torch_parallelize, "IS_MUSA_AVAILABLE", True)
    monkeypatch.setenv("VEOMNI_MUSA_DEEPEP_FSDP_STREAM_COMPAT", "1")
    monkeypatch.setenv("TORCH_MUSA_FSDP2_OVERLAP_LEVEL", "2")
    monkeypatch.setattr(
        torch,
        "musa",
        SimpleNamespace(
            current_stream=lambda: current_stream,
            Stream=lambda priority: created_streams.append(priority) or ("stream", len(created_streams)),
        ),
        raising=False,
    )
    monkeypatch.setattr(FSDPCommContext, "lazy_init", original_lazy_init)

    torch_parallelize._apply_musa_deepep_fsdp_stream_compat_patch()
    patched = custom_overlap_patch.comm_context_lazy_init
    context = SimpleNamespace()
    patched(context, "musa")

    assert context.all_gather_copy_in_stream is current_stream
    assert context.all_gather_stream == ("stream", 1)
    assert context.reduce_scatter_stream is context.all_gather_stream
    assert created_streams == [0]


def test_musa_deepep_fsdp_stream_compat_does_not_change_level_one(monkeypatch):
    from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPCommContext

    original_streams = (object(), object(), object())

    def original_lazy_init(context, _device):
        (
            context.all_gather_copy_in_stream,
            context.all_gather_stream,
            context.reduce_scatter_stream,
        ) = original_streams

    custom_overlap_patch = SimpleNamespace(
        comm_context_lazy_init=original_lazy_init,
        _FSDP2_OVERLAP_LEVEL=SimpleNamespace(value=1),
    )
    _install_fake_torch_musa(monkeypatch, custom_overlap_patch)
    monkeypatch.setattr(torch_parallelize, "IS_MUSA_AVAILABLE", True)
    monkeypatch.setenv("VEOMNI_MUSA_DEEPEP_FSDP_STREAM_COMPAT", "true")
    monkeypatch.setenv("TORCH_MUSA_FSDP2_OVERLAP_LEVEL", "2")
    monkeypatch.setattr(FSDPCommContext, "lazy_init", original_lazy_init)

    torch_parallelize._apply_musa_deepep_fsdp_stream_compat_patch()
    context = SimpleNamespace()
    custom_overlap_patch.comm_context_lazy_init(context, "musa")

    assert (
        context.all_gather_copy_in_stream,
        context.all_gather_stream,
        context.reduce_scatter_stream,
    ) == original_streams


def test_musa_deepep_fsdp_stream_compat_can_adopt_deepep_stream(monkeypatch):
    from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPCommContext

    current_stream = object()

    def original_lazy_init(context, _device):
        context.all_gather_copy_in_stream = object()
        context.all_gather_stream = object()
        context.reduce_scatter_stream = object()

    custom_overlap_patch = SimpleNamespace(
        comm_context_lazy_init=original_lazy_init,
        _FSDP2_OVERLAP_LEVEL=SimpleNamespace(value=2),
    )
    _install_fake_torch_musa(monkeypatch, custom_overlap_patch)
    monkeypatch.setattr(torch_parallelize, "IS_MUSA_AVAILABLE", True)
    monkeypatch.setattr(torch_parallelize, "_MUSA_DEEPEP_FSDP_COMM_CONTEXTS", [])
    monkeypatch.setattr(torch_parallelize, "_MUSA_DEEPEP_COMM_STREAM", None)
    monkeypatch.setattr(torch_parallelize, "_MUSA_DEEPEP_SHARED_STREAM_LOGGED", False)
    monkeypatch.setenv("VEOMNI_MUSA_DEEPEP_FSDP_STREAM_COMPAT", "1")
    monkeypatch.setenv("VEOMNI_MUSA_DEEPEP_FSDP_SHARED_COMM_STREAM", "1")
    monkeypatch.setenv("TORCH_MUSA_FSDP2_OVERLAP_LEVEL", "2")
    monkeypatch.setattr(
        torch,
        "musa",
        SimpleNamespace(current_stream=lambda: current_stream, Stream=lambda priority: ("temporary", priority)),
        raising=False,
    )
    monkeypatch.setattr(FSDPCommContext, "lazy_init", original_lazy_init)

    torch_parallelize._apply_musa_deepep_fsdp_stream_compat_patch()
    context = SimpleNamespace()
    custom_overlap_patch.comm_context_lazy_init(context, "musa")
    shared_stream = object()
    torch_parallelize._set_musa_deepep_fsdp_shared_comm_stream(shared_stream)

    assert context.all_gather_copy_in_stream is current_stream
    assert context.all_gather_stream is shared_stream
    assert context.reduce_scatter_stream is shared_stream
