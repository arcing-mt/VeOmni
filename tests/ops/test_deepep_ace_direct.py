import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from veomni.distributed.moe import deepep_ace_direct as lease


@pytest.mark.parametrize("enabled", [False, True])
def test_direct_write_uses_config_and_ignores_retired_environment(monkeypatch, enabled):
    config = SimpleNamespace(moe_deepep_combine_direct_write=enabled)
    monkeypatch.setattr(lease.ace, "get_ops_config", lambda: config)
    monkeypatch.setenv("VEOMNI_ACE_COMBINE_STAGING", "1")
    monkeypatch.setenv("VEOMNI_ACE_COMBINE_DIRECT_WRITE", str(int(not enabled)))
    assert lease.ace.combine_direct_write_enabled() is enabled
    assert lease.ace._ACEState(None, 32, 4).combine_direct_write is enabled


def test_direct_write_disabled_without_config(monkeypatch):
    monkeypatch.setattr(lease.ace, "get_ops_config", lambda: None)
    assert not lease.ace.combine_direct_write_enabled()


def test_buffer_cache_separates_single_and_double_windows(monkeypatch):
    created = []
    sizing = SimpleNamespace(get_nvl_buffer_size_hint=lambda *args: 0, get_rdma_buffer_size_hint=lambda *args: 0)

    class FakeBuffer:
        set_num_sms = staticmethod(lambda n: None)
        get_dispatch_config = staticmethod(lambda n: sizing)
        get_combine_config = staticmethod(lambda n: sizing)

        def __init__(self, group, nvl, rdma, **kwargs):
            self.slots = kwargs["num_ace_buffers"]
            created.append(self)

        def get_ace_combine_buffer(self, *args):
            pass

    monkeypatch.setattr(lease.ace, "_load_deepep", lambda: (FakeBuffer, None, None))
    monkeypatch.setattr(lease.ace, "_BUFFER_CACHE", {})
    monkeypatch.setattr(lease.ace, "_CAPACITY_CHECKED", set())
    monkeypatch.setattr(lease.ace, "get_ops_config", lambda: None)
    monkeypatch.setattr(lease.ace.logger, "info_rank0", lambda *args: None)
    monkeypatch.setattr(
        lease.ace.dist, "all_gather_object", lambda dst, obj, **kwargs: dst.__setitem__(slice(None), [obj, obj])
    )
    group = SimpleNamespace(size=lambda: 2)
    x = torch.ones(2, 8, dtype=torch.bfloat16)
    resolved = []
    for enabled in (False, True, False, True):
        config = SimpleNamespace(moe_deepep_combine_direct_write=enabled)
        monkeypatch.setattr(lease.ace, "get_ops_config", lambda config=config: config)
        resolved.append(lease.ace._ACEState(group, 32, 4)._get_buffer(x))
    assert [b.slots for b in resolved] == [1, 2, 1, 2]
    assert len(created) == 2 and resolved[0] is resolved[2] and resolved[1] is resolved[3]


def test_rank_mode_mismatch_fails_before_buffer_creation(monkeypatch):
    monkeypatch.setattr(lease.ace, "_load_deepep", lambda: (None, None, None))
    monkeypatch.setattr(lease.ace, "_CAPACITY_CHECKED", set())
    monkeypatch.setattr(lease.ace, "get_ops_config", lambda: SimpleNamespace(moe_deepep_combine_direct_write=True))
    monkeypatch.setattr(
        lease.ace.dist,
        "all_gather_object",
        lambda dst, obj, **kwargs: dst.__setitem__(slice(None), [obj, (8192, False)]),
    )
    with pytest.raises(RuntimeError, match="all EP ranks"):
        lease.ace._ACEState(SimpleNamespace(size=lambda: 2), 32, 4)._get_buffer(torch.ones(2, 8))


class Event:
    def __init__(self, log, label):
        self.log, self.label = log, label

    def current_stream_wait(self):
        self.log.append(self.label)


class Buffer:
    def __init__(self, log):
        self._veomni_combine_shape = (4, 8, 2)
        self.slots = [torch.empty(8, 8, dtype=torch.bfloat16) for _ in range(2)]
        self.ps = [torch.empty(8, 2) for _ in range(2)]
        self.log = log

    def get_ace_combine_buffer(self, rows, hidden, topk, weights, buffer_index):
        self.log.append(("slot", buffer_index))
        return self.slots[buffer_index][: max(1, rows)], self.ps[buffer_index][: max(1, rows)] if weights else None

    def combine(self, x, handle, **kwargs):
        self.log.append("submit")
        return x.clone(), None, Event(self.log, "complete")


def test_pool_waits_before_writer_and_cycles(monkeypatch):
    log = []
    b = Buffer(log)
    pool = lease.WindowPool(b)
    monkeypatch.setattr(lease.ace, "_current_stream_event", lambda: log.append("producer"))
    state = type("State", (), {"handle": ()})()
    x = torch.ones(2, 8, dtype=torch.bfloat16)
    saved = []
    for i in range(5):
        before = len(log)

        def write(dst, p, i=i):
            log.append("write")
            dst.fill_(i)

        saved.append(pool.produce_and_combine(state, x, 2, write, False)[0])
        expected = (["complete"] if i >= 2 else []) + [("slot", i % 2), "write", "producer", "submit", "complete"]
        assert log[before:] == expected
    assert all(torch.equal(t, torch.full_like(t, i)) for i, t in enumerate(saved))


def test_empty_preserves_slot_one_pointer(monkeypatch):
    log = []
    b = Buffer(log)
    pool = lease.WindowPool(b)
    pool.next = 1
    monkeypatch.setattr(lease.ace, "_current_stream_event", lambda: None)
    state = type("State", (), {"handle": ()})()
    x = torch.empty(0, 8, dtype=torch.bfloat16)

    def write(*args):
        raise AssertionError("zero-row producer must not launch")

    result = pool.produce_and_combine(state, x, 0, write, True)
    assert ("slot", 1) in log and result[0].data_ptr() != 0


def test_exception_poison_stops_reuse(monkeypatch):
    monkeypatch.setattr(lease.ace, "_current_stream_event", lambda: None)
    log = []
    pool = lease.WindowPool(Buffer(log))
    state = type("State", (), {"handle": ()})()
    x = torch.ones(2, 8, dtype=torch.bfloat16)

    def fail(*args):
        raise ValueError("writer failed")

    with pytest.raises(ValueError):
        pool.produce_and_combine(state, x, 2, fail, False)
    with pytest.raises(RuntimeError, match="previous"):
        pool.produce_and_combine(state, x, 2, fail, False)


def test_checkpoint_early_stop_records_writer_event(monkeypatch):
    log = []
    pool = lease.WindowPool(Buffer(log))
    state = type("State", (), {"handle": ()})()
    x = torch.ones(2, 8, dtype=torch.bfloat16)
    monkeypatch.setattr(lease.ace, "_current_stream_event", lambda: Event(log, "writer_done"))

    class StopRecompute(Exception):
        pass

    with pytest.raises(StopRecompute):
        with pool.acquire(state, x, 2, False) as loan:
            loan.x.fill_(7)
            raise StopRecompute()
    assert not pool.poisoned and "submit" not in log
    pool.next = 0
    with pool.acquire(state, x, 2, False):
        assert log[-2:] == ["writer_done", ("slot", 0)]


def test_window_does_not_retain_expert_graph_until_cyclic_gc(monkeypatch):
    pool = lease.WindowPool(Buffer([]))
    monkeypatch.setattr(lease.ace, "_current_stream_event", lambda: Event([], "producer"))

    class Restore(torch.autograd.Function):
        @staticmethod
        def forward(ctx, weighted, loan):
            loan.x.copy_(weighted)
            return loan.x

        @staticmethod
        def backward(ctx, grad):
            return grad, None

    class Combine(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, loan):
            return loan.submit(x)[0]

        @staticmethod
        def backward(ctx, grad):
            return grad, None

    def invoke():
        state = SimpleNamespace(handle=())
        hidden = torch.ones(2, 8, dtype=torch.bfloat16, requires_grad=True)
        expert = hidden * 2
        weighted = expert * expert
        with pool.acquire(state, expert, 2, False) as loan:
            window = Restore.apply(weighted, loan)
            output = Combine.apply(window, loan)
        assert output.data_ptr() != window.data_ptr()
        return weakref.ref(window), weakref.ref(expert)

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        refs = invoke()
        assert all(ref() is None for ref in refs), "window/producer graph requires cyclic GC to release"
    finally:
        if was_enabled:
            gc.enable()
        gc.collect()
