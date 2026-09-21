import torch
from torch.distributed import ReduceOp

from veomni.ops.platform.musa.mccl_premul_sum import (
    _wrap_custom_overlap_reduce_scatter,
    mccl_reduce_op_wrapper,
)


class _Handle:
    def __init__(self):
        self.wait_calls = 0

    def wait(self):
        self.wait_calls += 1


def test_mccl_wrapper_preserves_async_sum():
    handle = _Handle()

    def collective(*args, **kwargs):
        assert kwargs["op"] is ReduceOp.SUM
        assert kwargs["async_op"] is True
        return handle

    wrapper = mccl_reduce_op_wrapper(collective, "tensor", op_arg_index=1, group_arg_index=2)
    tensor = torch.ones(2)
    result = wrapper(tensor, op=ReduceOp.SUM, async_op=True)

    assert result is handle
    assert handle.wait_calls == 0
    assert torch.equal(tensor, torch.ones(2))


def test_mccl_wrapper_waits_before_scaling_premul_sum():
    handle = _Handle()
    factor = 0.5

    class MockPremulSum:
        def __getstate__(self):
            return (ReduceOp.PREMUL_SUM.__getstate__(), factor)

    def collective(*args, **kwargs):
        assert kwargs["op"] is ReduceOp.SUM
        args[0].mul_(4)
        return handle

    wrapper = mccl_reduce_op_wrapper(collective, "tensor", op_arg_index=1, group_arg_index=2)
    tensor = torch.ones(2)
    result = wrapper(tensor, op=MockPremulSum(), async_op=True)

    assert result is handle
    assert handle.wait_calls == 1
    assert torch.equal(tensor, torch.full_like(tensor, 2))


def test_custom_overlap_wrapper_translates_premul_sum():
    handle = _Handle()
    factor = 0.25
    calls = []

    class MockPremulSum:
        def __getstate__(self):
            return (ReduceOp.PREMUL_SUM.__getstate__(), factor)

    class Comm:
        def __call__(
            self,
            output_tensor,
            input_tensor,
            group,
            op,
            async_op=False,
        ):
            calls.append((group, op, async_op))
            assert op is ReduceOp.SUM
            output_tensor.mul_(8)
            return handle

    comm = Comm()
    wrapped_call = _wrap_custom_overlap_reduce_scatter(Comm.__call__)
    output = torch.ones(2)
    result = wrapped_call(
        comm,
        output_tensor=output,
        input_tensor=torch.ones(2),
        group="group",
        op=MockPremulSum(),
    )

    assert result is handle
    assert calls == [("group", ReduceOp.SUM, False)]
    assert handle.wait_calls == 1
    assert torch.equal(output, torch.full_like(output, 2))


def test_custom_overlap_wrapper_preserves_avg():
    calls = []

    class Comm:
        def __call__(
            self,
            output_tensor,
            input_tensor,
            group,
            op,
            async_op=False,
        ):
            calls.append(op)
            output_tensor.mul_(2)

    comm = Comm()
    wrapped_call = _wrap_custom_overlap_reduce_scatter(Comm.__call__)
    output = torch.ones(2)
    wrapped_call(
        comm,
        output_tensor=output,
        input_tensor=torch.ones(2),
        group="group",
        op=ReduceOp.AVG,
    )

    assert calls == [ReduceOp.AVG]
    assert torch.equal(output, torch.full_like(output, 2))


def test_mccl_wrapper_preserves_avg():
    """MCCL handles AVG natively, so the wrapper must not rewrite it or scale after the fact.

    The AVG -> SUM + ``mul_(1 / group_size)`` rewrite used to cost an extra elementwise
    kernel and a ``handle.wait()`` per reduce-scatter.  Regressing this would silently
    bring both back, so pin the pass-through.
    """

    class _Group:
        def size(self):
            return 2  # so a resurrected mul_(1 / group_size) would actually change the value

    handle = _Handle()
    calls = []

    def collective(*args, **kwargs):
        calls.append(kwargs["op"])
        args[0].mul_(2)
        return handle

    wrapper = mccl_reduce_op_wrapper(collective, "tensor", op_arg_index=1, group_arg_index=2)
    tensor = torch.ones(2)
    result = wrapper(tensor, op=ReduceOp.AVG, group=_Group(), async_op=True)

    assert calls == [ReduceOp.AVG], "AVG must reach the collective unchanged"
    assert result is handle
    assert handle.wait_calls == 0, "an untouched AVG must not wait"
    assert torch.equal(tensor, torch.full_like(tensor, 2)), "scaling by 1/1 would hide a resurrected mul_"


def test_mccl_wrapper_rewrites_avg_only_when_native_is_disabled(monkeypatch):
    """The escape hatch must restore the old rewrite for a build whose MCCL rejects AVG.

    Passes ``op`` and ``group`` positionally on purpose: the wrapper's positional branch is
    otherwise never exercised, and a wrong ``op_arg_index``/``group_arg_index`` would ship.
    """
    monkeypatch.setenv("VEOMNI_MCCL_NATIVE_AVG", "0")
    handle = _Handle()
    calls = []

    class _Group:
        def size(self):
            return 2

    def collective(*args, **kwargs):
        # `op` and `group` arrive positionally here, so read whichever way they were passed.
        calls.append(kwargs["op"] if "op" in kwargs else args[1])
        args[0].mul_(2)  # stands in for summing two ranks that each held 1
        return handle

    wrapper = mccl_reduce_op_wrapper(collective, "tensor", op_arg_index=1, group_arg_index=2)
    tensor = torch.ones(2)
    result = wrapper(tensor, ReduceOp.AVG, _Group(), async_op=True)

    assert calls == [ReduceOp.SUM], "the escape hatch must put SUM on the wire"
    assert result is handle
    assert handle.wait_calls == 1
    assert torch.equal(tensor, torch.ones(2)), "1 * 2 (sum) * (1 / 2) == 1"

    monkeypatch.setenv("VEOMNI_MCCL_NATIVE_AVG", "1")
    calls.clear()
    wrapper(tensor, ReduceOp.AVG, _Group(), async_op=True)
    assert calls == [ReduceOp.AVG]
