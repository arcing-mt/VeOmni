import torch
from torch.distributed import ReduceOp

from veomni.ops.platform.musa.mccl_premul_sum import mccl_reduce_op_wrapper


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
