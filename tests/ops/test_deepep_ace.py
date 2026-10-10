from types import SimpleNamespace

import pytest
import torch

from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.distributed.moe import deepep_ace
from veomni.distributed.moe.deepep_ace import (
    _ACTIVE_SHARED_EXPERT,
    _ACEState,
    _compact_permute,
    _compact_unpermute,
    _load_deepep,
    _SharedExpertOverlap,
)


def test_deepep_ace_compact_round_trip_cpu():
    recv_hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    recv_indices = torch.tensor([[1, 0], [-1, 1], [0, -1]], dtype=torch.int64)
    recv_probs = torch.tensor([[0.25, 0.75], [0.0, 1.0], [1.0, 0.0]])

    permuted, probs, rows, counts = _compact_permute(recv_hidden, recv_indices, recv_probs, num_local_experts=2)
    _, _, _, host_counts = _compact_permute(
        recv_hidden, recv_indices, recv_probs, num_local_experts=2, expert_counts=[2, 2]
    )
    restored = _compact_unpermute(permuted, probs, rows, recv_hidden.shape[0])

    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert counts.tolist() == [2, 2]
    assert host_counts.tolist() == [2, 2]
    assert torch.equal(restored, expected)


def test_deepep_ace_compact_unpermute_backward_cpu():
    expert_outputs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    probs = torch.tensor([0.25, 0.5, 0.75], requires_grad=True)
    token_rows = torch.tensor([0, 0, 1], dtype=torch.long)

    restored = _compact_unpermute(expert_outputs, probs, token_rows, recv_tokens=2)
    restored.sum().backward()

    assert torch.equal(expert_outputs.grad, probs.detach().unsqueeze(-1).expand_as(expert_outputs))
    assert torch.equal(probs.grad, expert_outputs.detach().sum(dim=-1))


@pytest.mark.skipif(not hasattr(torch, "musa") or not torch.musa.is_available(), reason="requires MUSA")
def test_deepep_ace_musa_counting_sort_matches_stable_argsort():
    from veomni.ops.kernels.moe.musa_deepep_compact import musa_deepep_stable_slots

    generator = torch.Generator().manual_seed(17)
    num_tokens, top_k, num_experts, num_assignments = 257, 8, 32, 1500
    flat_cpu = torch.full((num_tokens * top_k,), -1, dtype=torch.long)
    valid_slots = torch.randperm(flat_cpu.numel(), generator=generator)[:num_assignments].sort().values
    expert_ids = torch.randint(0, num_experts, (num_assignments,), generator=generator)
    flat_cpu[valid_slots] = expert_ids

    flat = flat_cpu.musa()
    counts = torch.bincount(expert_ids, minlength=num_experts).musa()
    actual_slots, actual_rows = musa_deepep_stable_slots(flat, counts, top_k, num_assignments)

    reference_valid = torch.nonzero(flat >= 0, as_tuple=False).flatten()
    reference_order = torch.argsort(flat.index_select(0, reference_valid), stable=True)
    reference_slots = reference_valid.index_select(0, reference_order)
    reference_rows = torch.div(reference_slots, top_k, rounding_mode="floor")

    assert torch.equal(actual_slots, reference_slots)
    assert torch.equal(actual_rows, reference_rows)


@pytest.mark.skipif(not hasattr(torch, "musa") or not torch.musa.is_available(), reason="requires MUSA")
def test_deepep_ace_musa_counting_sort_handles_empty_receive():
    from veomni.ops.kernels.moe.musa_deepep_compact import musa_deepep_stable_slots

    flat = torch.empty(0, dtype=torch.long, device="musa")
    counts = torch.zeros(32, dtype=torch.long, device="musa")
    slots, rows = musa_deepep_stable_slots(flat, counts, top_k=8, num_assignments=0)

    assert slots.shape == (0,)
    assert rows.shape == (0,)


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ([2, 1], [1, 2]),
        ([0, 0], [1, 0]),
    ],
)
def test_deepep_ace_counting_sort_rejects_mismatched_counts(actual, expected):
    from veomni.ops.kernels.moe.musa_deepep_compact import _require_matching_counts

    with pytest.raises(RuntimeError, match="expert counts do not match"):
        _require_matching_counts(torch.tensor(actual), torch.tensor(expected))


def test_deepep_ace_dispatch_uses_caller_previous_event(monkeypatch):
    previous_event = object()
    layout_event = object()

    class FakeGroup:
        def size(self):
            return 2

    class FakeBuffer:
        def get_dispatch_layout(self, *args, **kwargs):
            assert kwargs["previous_event"] is previous_event
            return None, None, None, None, layout_event

        def dispatch(self, hidden_states, **kwargs):
            assert kwargs["previous_event"] is layout_event
            return hidden_states, kwargs["topk_idx"], kwargs["topk_weights"], [1], object(), object()

    state = _ACEState(FakeGroup(), num_experts=2, top_k=1)
    monkeypatch.setattr(state, "_get_buffer", lambda _hidden_states: FakeBuffer())
    hidden_states = torch.ones(1, 2)
    selected_experts = torch.zeros(1, 1, dtype=torch.long)
    routing_weights = torch.ones(1, 1)

    received, _, _ = state.dispatch(hidden_states, selected_experts, routing_weights, previous_event=previous_event)
    assert received is hidden_states


def test_deepep_ace_dispatch_extends_external_event_through_payload_copies(monkeypatch):
    order = []
    contiguous, as_float = torch.Tensor.contiguous, torch.Tensor.float

    def prepare_hidden(tensor, *args, **kwargs):
        order.append("contiguous")
        return contiguous(tensor, *args, **kwargs)

    def prepare_probs(tensor, *args, **kwargs):
        order.append("float")
        return as_float(tensor, *args, **kwargs)

    class PreviousEvent:
        def current_stream_wait(self):
            order.append("wait_external")

    ready_event, layout_event = object(), object()

    def capture_ready():
        order.append("capture_ready")
        assert order == ["wait_external", "contiguous", "float", "capture_ready"]
        return ready_event

    class FakeBuffer:
        def get_dispatch_layout(self, *args, **kwargs):
            assert kwargs["previous_event"] is ready_event
            return None, None, None, None, layout_event

        def dispatch(self, hidden_states, **kwargs):
            assert hidden_states.is_contiguous()
            assert kwargs["topk_weights"].dtype == torch.float32
            assert kwargs["previous_event"] is layout_event
            return hidden_states, kwargs["topk_idx"], kwargs["topk_weights"], [1], object(), object()

    state = _ACEState(SimpleNamespace(size=lambda: 2), num_experts=2, top_k=1)
    hidden = torch.ones(2, 3).t()
    indices = torch.zeros(3, 1, dtype=torch.long)
    probs = torch.ones(3, 1, dtype=torch.bfloat16)
    monkeypatch.setattr(torch.Tensor, "contiguous", prepare_hidden)
    monkeypatch.setattr(torch.Tensor, "float", prepare_probs)
    monkeypatch.setattr(state, "_get_buffer", lambda _hidden: FakeBuffer())
    monkeypatch.setattr(deepep_ace, "_current_stream_event", capture_ready)
    state.dispatch(hidden, indices, probs, previous_event=PreviousEvent())


def test_deepep_overlap_captures_ready_after_payload_preparation(monkeypatch):
    order = []
    contiguous, as_float = torch.Tensor.contiguous, torch.Tensor.float

    def prepare_hidden(tensor, *args, **kwargs):
        order.append("contiguous")
        return contiguous(tensor, *args, **kwargs)

    def prepare_probs(tensor, *args, **kwargs):
        order.append("float")
        return as_float(tensor, *args, **kwargs)

    ready_event = object()

    def capture_ready():
        order.append("capture_ready")
        assert order == ["contiguous", "float", "capture_ready"]
        return ready_event

    class StopAfterDispatch(RuntimeError):
        pass

    def inspect_dispatch(hidden, indices, probs, state):
        assert hidden.is_contiguous()
        assert probs.dtype == torch.float32
        assert state.previous_event is ready_event
        raise StopAfterDispatch

    group = SimpleNamespace(size=lambda: 2)
    ep_class = object()
    monkeypatch.setattr(deepep_ace, "get_parallel_state", lambda: SimpleNamespace(ep_enabled=True, ep_group=group))
    monkeypatch.setattr(deepep_ace, "get_ops_config", lambda: SimpleNamespace(moe_dispatcher="deepep_ace"))
    monkeypatch.setattr(deepep_ace, "_supported_ep_classes", lambda: (ep_class,))
    monkeypatch.setattr(deepep_ace, "_ACTIVE_SHARED_EXPERT", SimpleNamespace(get=lambda: object()))
    monkeypatch.setattr(deepep_ace, "_current_stream_event", capture_ready)
    monkeypatch.setattr(deepep_ace._ACEDispatch, "apply", inspect_dispatch)
    monkeypatch.setattr(torch.Tensor, "contiguous", prepare_hidden)
    monkeypatch.setattr(torch.Tensor, "float", prepare_probs)
    with pytest.raises(StopAfterDispatch):
        deepep_ace.dispatch_to_ep_class_deepep(
            ep_class,
            2,
            torch.ones(3, 1, dtype=torch.bfloat16),
            torch.zeros(3, 1, dtype=torch.long),
            torch.ones(2, 3).t(),
        )


def test_deepep_ace_is_a_dispatcher_selection():
    config = OpsImplementationConfig(moe_dispatcher="deepep_ace")
    assert config.moe_implementation == "fused_triton"
    assert config.moe_dispatcher == "deepep_ace"
    assert config.moe_deepep_num_sms == 20
    assert config.moe_deepep_token_capacity == 8192


def test_standard_deepep_is_a_dispatcher_selection():
    config = OpsImplementationConfig(moe_dispatcher="deepep", moe_shared_expert_overlap=True)
    assert config.moe_dispatcher == "deepep"


def test_standard_deepep_constructs_buffer_without_ace_workspace(monkeypatch):
    class FakeGroup:
        def size(self):
            return 2

    class FakeSizeHint:
        def get_nvl_buffer_size_hint(self, *_args):
            return 1024

        def get_rdma_buffer_size_hint(self, *_args):
            return 0

    class FakeBuffer:
        num_sms = None

        @classmethod
        def set_num_sms(cls, value):
            cls.num_sms = value

        @staticmethod
        def get_dispatch_config(_group_size):
            return FakeSizeHint()

        @staticmethod
        def get_combine_config(_group_size):
            return FakeSizeHint()

        def __init__(
            self,
            group,
            nvl_bytes,
            rdma_bytes,
            use_ace=False,
            num_ace_buffers=1,
            token_num=0,
            hidden_size=0,
            num_topk=0,
        ):
            self.args = (group, nvl_bytes, rdma_bytes)
            self.use_ace = use_ace
            self.workspace = (num_ace_buffers, token_num, hidden_size, num_topk)

    monkeypatch.setattr(deepep_ace, "_load_deepep", lambda: (FakeBuffer, object, object))
    monkeypatch.setattr(
        deepep_ace,
        "get_ops_config",
        lambda: SimpleNamespace(moe_deepep_num_sms=20, moe_deepep_token_capacity=8192),
    )
    state = _ACEState(FakeGroup(), num_experts=4, top_k=2, use_ace=False)
    buffer = state._resolve_buffer(torch.ones(3, 8))

    assert FakeBuffer.num_sms == 20
    assert buffer.use_ace is False
    assert buffer.workspace == (1, 0, 0, 0)


@pytest.mark.parametrize("use_ace", [False, True])
def test_deepep_shared_fsdp_stream_requires_stream_accessor(monkeypatch, use_ace):
    buffer = object()
    state = _ACEState(object(), num_experts=4, top_k=2, use_ace=use_ace)
    monkeypatch.setattr(state, "_resolve_buffer", lambda _hidden_states: buffer)
    monkeypatch.setenv("VEOMNI_MUSA_DEEPEP_FSDP_SHARED_COMM_STREAM", "true")
    monkeypatch.setattr(deepep_ace, "_FSDP_SHARED_STREAM_REGISTERED_BUFFERS", set())

    for _ in range(2):
        with pytest.raises(RuntimeError, match="get_comm_stream"):
            state._get_buffer(torch.ones(3, 8))
        assert state._buffer is None


def test_deepep_shared_fsdp_stream_registers_each_buffer_once(monkeypatch):
    from veomni.distributed import torch_parallelize

    stream = object()
    buffer = SimpleNamespace(get_comm_stream=lambda: stream)
    registered = []
    monkeypatch.setenv("VEOMNI_MUSA_DEEPEP_FSDP_SHARED_COMM_STREAM", "true")
    monkeypatch.setattr(deepep_ace, "_FSDP_SHARED_STREAM_REGISTERED_BUFFERS", set())
    monkeypatch.setattr(torch_parallelize, "_set_musa_deepep_fsdp_shared_comm_stream", registered.append)

    for _ in range(2):
        state = _ACEState(object(), num_experts=4, top_k=2, use_ace=False)
        monkeypatch.setattr(state, "_resolve_buffer", lambda _hidden_states: buffer)
        assert state._get_buffer(torch.ones(3, 8)) is buffer
        assert state._get_buffer(torch.ones(3, 8)) is buffer
    assert registered == [stream]


def test_deepep_shared_fsdp_stream_rejects_missing_stream(monkeypatch):
    state = _ACEState(object(), num_experts=4, top_k=2, use_ace=False)
    monkeypatch.setattr(state, "_resolve_buffer", lambda _hidden_states: SimpleNamespace(get_comm_stream=lambda: None))
    monkeypatch.setenv("VEOMNI_MUSA_DEEPEP_FSDP_SHARED_COMM_STREAM", "true")
    monkeypatch.setattr(deepep_ace, "_FSDP_SHARED_STREAM_REGISTERED_BUFFERS", set())

    with pytest.raises(RuntimeError, match="returned no communication stream"):
        state._get_buffer(torch.ones(3, 8))
    assert state._buffer is None


def test_deepep_without_shared_fsdp_stream_accepts_legacy_buffer(monkeypatch):
    buffer = object()
    state = _ACEState(object(), num_experts=4, top_k=2, use_ace=False)
    monkeypatch.setattr(state, "_resolve_buffer", lambda _hidden_states: buffer)
    monkeypatch.setenv("VEOMNI_MUSA_DEEPEP_FSDP_SHARED_COMM_STREAM", "false")

    assert state._get_buffer(torch.ones(3, 8)) is buffer


def test_shared_expert_overlap_rejects_non_deepep_dispatcher():
    with pytest.raises(ValueError, match="requires a DeepEP dispatcher"):
        OpsImplementationConfig(moe_dispatcher="alltoall", moe_shared_expert_overlap=True)


def test_deepep_ace_num_sms_must_be_even_and_positive():
    with pytest.raises(ValueError, match="positive even"):
        OpsImplementationConfig(moe_deepep_num_sms=3)


def test_deepep_ace_token_capacity_must_be_positive():
    with pytest.raises(ValueError, match="token_capacity must be positive"):
        OpsImplementationConfig(moe_deepep_token_capacity=0)


def test_deepep_wheel_event_exports_are_resolved():
    _, event_handle, event_overlap = _load_deepep()
    assert event_handle is not None
    assert event_overlap is not None


def test_shared_expert_overlap_runs_inline_and_joins(monkeypatch):
    """The overlap must gate and compute the shared expert without a stream of its own.

    An earlier revision queued it on a dedicated MUSA stream; that cost more
    than the shared expert itself (~0.7 s/step on Qwen3.5-35B-A3B).  Pin the
    contract as well as the numerics: no extra stream, no cross-stream event,
    and `finish` returns exactly what `start` computed.
    """
    stream = object()
    monkeypatch.setattr(torch, "musa", SimpleNamespace(current_stream=lambda: stream), raising=False)
    created_streams = []
    monkeypatch.setattr(torch, "Stream", lambda *a, **k: created_streams.append((a, k)), raising=False)

    hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    class FakeSharedExpert(torch.nn.Module):
        def forward(self, x):
            return x * 2.0

    class FakeGate(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 1)

    overlap = _SharedExpertOverlap(FakeSharedExpert(), FakeGate(), hidden_states)
    with pytest.raises(RuntimeError, match="was not started"):
        overlap.finish()

    overlap.start()
    expected = torch.sigmoid(torch.zeros(hidden_states.shape[0], 1)) * (hidden_states * 2.0)

    assert overlap.finish() is overlap.output
    assert torch.allclose(overlap.output, expected)
    assert overlap.stream is stream
    assert created_streams == []
    assert not hasattr(overlap, "event")

    with pytest.raises(RuntimeError, match="already started"):
        overlap.start()
    other = object()
    monkeypatch.setattr(torch, "musa", SimpleNamespace(current_stream=lambda: other), raising=False)
    with pytest.raises(RuntimeError, match="different stream"):
        overlap.finish()


def test_shared_expert_is_issued_between_dispatch_and_wait(monkeypatch):
    """The overlap only exists if the shared expert is issued inside the dispatch window.

    A shared expert issued before ``buffer.dispatch`` cannot hide under the
    payload; one issued after ``wait_dispatch`` does not overlap at all.  Both
    mistakes have happened in this file's history, so assert the order against
    a fake DeepEP buffer rather than trusting the call site.
    """
    order: list[str] = []

    class FakeEvent:
        def current_stream_wait(self):
            pass

    stream = object()
    monkeypatch.setattr(torch, "musa", SimpleNamespace(current_stream=lambda: stream), raising=False)
    monkeypatch.setattr(deepep_ace, "_current_stream_event", FakeEvent)

    class FakeGroup:
        def size(self):
            return 2

    class FakeBuffer:
        def get_dispatch_layout(self, *args, **kwargs):
            order.append("layout")
            return None, None, None, None, FakeEvent()

        def dispatch(self, hidden_states, **kwargs):
            order.append("dispatch")
            return hidden_states, kwargs["topk_idx"], kwargs["topk_weights"], [1, 1], object(), FakeEvent()

    state = _ACEState(FakeGroup(), num_experts=2, top_k=1)
    monkeypatch.setattr(state, "_get_buffer", lambda _hidden_states: FakeBuffer())

    class FakeSharedExpert(torch.nn.Module):
        def forward(self, x):
            order.append("shared_expert")
            return x * 2.0

    class FakeGate(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 1)

    hidden_states = torch.tensor([[1.0, 2.0], [1.0, 1.0]])
    selected_experts = torch.tensor([[0], [1]])
    routing_weights = torch.ones(2, 1)
    overlap = _SharedExpertOverlap(FakeSharedExpert(), FakeGate(), hidden_states)

    def run_dispatch(active):
        order.clear()
        token = _ACTIVE_SHARED_EXPERT.set(active) if active is not None else None
        try:
            state.dispatch(hidden_states, selected_experts, routing_weights)
        finally:
            if token is not None:
                _ACTIVE_SHARED_EXPERT.reset(token)

    # Without an active overlap context the dispatcher must not run a shared expert.
    run_dispatch(None)
    assert order == ["layout", "dispatch"], order

    # With one, it is issued after the dispatch is submitted ...
    run_dispatch(overlap)
    assert order == ["layout", "dispatch", "shared_expert"], order

    # ... and before the payload is awaited: `wait_dispatch` must never observe a
    # missing shared expert, otherwise the overlap is silently gone.
    def wait():
        assert order[-1] == "shared_expert", f"shared expert not issued before wait_dispatch: {order}"
        order.append("wait_dispatch")
        return state.dispatch_event.current_stream_wait()

    monkeypatch.setattr(state, "wait_dispatch", wait)
    state.wait_dispatch()
    assert order[-1] == "wait_dispatch"
