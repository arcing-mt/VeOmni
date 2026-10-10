"""Alignment preserves rank blocks, operation and asynchronous copy ordering."""

from types import SimpleNamespace

import pytest
import torch
from torch.distributed.fsdp._fully_shard import _fsdp_collectives as collectives

from veomni.distributed.fsdp2 import musa_collective_padding as padding


@pytest.mark.parametrize("gather", [True, False])
@pytest.mark.parametrize("async_op", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_padding_preserves_collective_payload(monkeypatch, gather, async_op, dtype):
    n, world, padded = 7, 8, 8
    group = SimpleNamespace(size=lambda: world)
    order = []
    pending = []

    def wait():
        order.append("wait")
        for write in pending:
            write()
        pending.clear()

    work = SimpleNamespace(wait=wait)
    operation = object()
    monkeypatch.setattr(padding, "_eligible", lambda *a: (n, padded))

    def all_gather(self, output, source, group_arg, async_arg):
        assert group_arg is group and async_arg is async_op
        assert source.shape == (padded,) and source[-1] == 0

        def write():
            for rank in range(world):
                output.view(world, padded)[rank].copy_(source + rank)

        if async_arg:
            pending.append(write)
        else:
            write()
        order.append("collective")
        return work if async_arg else None

    def reduce_scatter(self, output, source, group_arg, op_arg, async_arg):
        assert group_arg is group and async_arg is async_op and op_arg is operation
        assert source.shape == (world * padded,)
        assert torch.count_nonzero(source.view(world, padded)[:, -1]) == 0

        def write():
            output.copy_(source.view(world, padded).sum(0))

        if async_arg:
            pending.append(write)
        else:
            write()
        order.append("collective")
        return work if async_arg else None

    monkeypatch.setattr(collectives.DefaultAllGather, "__call__", all_gather)
    monkeypatch.setattr(collectives.DefaultReduceScatter, "__call__", reduce_scatter)
    if gather:
        source = torch.arange(n, dtype=dtype)
        output = torch.empty(world * n, dtype=dtype)
        actual = padding.MUSAAlignedAllGather()(output, source, group, async_op)
        expected = torch.cat([source + rank for rank in range(world)])
    else:
        source = torch.arange(world * n, dtype=dtype)
        output = torch.empty(n, dtype=dtype)
        actual = padding.MUSAAlignedReduceScatter()(output, source, group, operation, async_op)
        expected = source.view(world, n).sum(0)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert actual is (work if async_op else None)
    assert order == (["collective", "wait"] if async_op else ["collective"])


def test_cpu_and_disabled_models_keep_default(monkeypatch):
    assert padding._eligible(torch.empty(8), torch.empty(1), SimpleNamespace(size=lambda: 8), True) is None
    monkeypatch.delenv("VEOMNI_MUSA_FSDP_SHARD_PADDING", raising=False)
    # Disabled configuration must not even enumerate or replace model collectives.
    padding.configure_musa_collective_padding(object())


@pytest.mark.parametrize("setting", ["TORCH_MUSA_FSDP2_OVERLAP_LEVEL", "TORCH_MUSA_FSDP2_COMM_TYPE"])
def test_reject_replaced_collective_configuration(monkeypatch, setting):
    monkeypatch.setenv("VEOMNI_MUSA_FSDP_SHARD_PADDING", "1")
    monkeypatch.setenv("TORCH_MUSA_FSDP2_OVERLAP_LEVEL", "0")
    monkeypatch.setenv("TORCH_MUSA_FSDP2_COMM_TYPE", "0")
    monkeypatch.setenv(setting, "2")
    with pytest.raises(RuntimeError, match="OVERLAP_LEVEL=0 and COMM_TYPE=0"):
        padding.configure_musa_collective_padding(object())


def test_enabled_install_is_model_local(monkeypatch):
    class Shard:
        def set_custom_all_gather(self, value):
            self.ag = value

        def set_custom_reduce_scatter(self, value):
            self.rs = value

    one, two, unrelated = Shard(), Shard(), Shard()
    monkeypatch.setattr(padding, "FSDPModule", Shard)
    monkeypatch.setenv("VEOMNI_MUSA_FSDP_SHARD_PADDING", "1")
    monkeypatch.setenv("TORCH_MUSA_FSDP2_OVERLAP_LEVEL", "0")
    monkeypatch.setenv("TORCH_MUSA_FSDP2_COMM_TYPE", "0")
    padding.configure_musa_collective_padding(SimpleNamespace(modules=lambda: [one, two, object()]))
    assert isinstance(one.ag, padding.MUSAAlignedAllGather)
    assert isinstance(two.rs, padding.MUSAAlignedReduceScatter)
    assert not hasattr(unrelated, "ag")


@pytest.mark.parametrize("explicit_zero", [False, True])
def test_legacy_overlap_obeys_explicit_level_priority(monkeypatch, explicit_zero):
    monkeypatch.setenv("VEOMNI_MUSA_FSDP_SHARD_PADDING", "1")
    monkeypatch.setenv("TORCH_MUSA_FSDP2_COMM_TYPE", "0")
    monkeypatch.setenv("TORCH_MUSA_FSDP2_DISABLE_OVERLAP", "0")
    monkeypatch.delenv("TORCH_MUSA_FSDP2_OVERLAP_LEVEL", raising=False)
    monkeypatch.setattr(padding, "IS_MUSA_AVAILABLE", False)
    model = SimpleNamespace(modules=lambda: [])
    if explicit_zero:
        monkeypatch.setenv("TORCH_MUSA_FSDP2_OVERLAP_LEVEL", "0")
        padding.configure_musa_collective_padding(model)
    else:
        with pytest.raises(RuntimeError, match="OVERLAP_LEVEL=0 and COMM_TYPE=0"):
            padding.configure_musa_collective_padding(model)
