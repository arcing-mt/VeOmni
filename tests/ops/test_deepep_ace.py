from types import SimpleNamespace

import pytest
import torch

from veomni.arguments.arguments_types import OpsImplementationConfig
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

    permuted, probs, rows, counts, row_map, width = _compact_permute(
        recv_hidden, recv_indices, recv_probs, num_local_experts=2
    )
    _, _, _, host_counts, _, _ = _compact_permute(
        recv_hidden, recv_indices, recv_probs, num_local_experts=2, expert_counts=[2, 2]
    )
    restored = _compact_unpermute(permuted, probs, rows, recv_hidden.shape[0], row_map=row_map, width=width)

    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert counts.tolist() == [2, 2]
    assert host_counts.tolist() == [2, 2]
    assert torch.equal(restored, expected)
    # The CPU path is driven by ``token_rows`` (row -> received token) with the
    # top-k slot count as its width; the expert-major map is a TE-path artifact.
    assert row_map is None and width == 2
    assert rows.tolist() == [0, 2, 0, 1]


def test_deepep_ace_compact_unpermute_backward_cpu():
    expert_outputs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)
    probs = torch.tensor([0.25, 0.5, 0.75], requires_grad=True)
    token_rows = torch.tensor([0, 0, 1], dtype=torch.long)

    restored = _compact_unpermute(expert_outputs, probs, token_rows, recv_tokens=2)
    restored.sum().backward()

    assert torch.equal(expert_outputs.grad, probs.detach().unsqueeze(-1).expand_as(expert_outputs))
    assert torch.equal(probs.grad, expert_outputs.detach().sum(dim=-1))


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


@pytest.mark.parametrize("setting", [None, "0", "1"])
def test_payload_ready_opt_in_orders_copies_before_communication(monkeypatch, setting):
    from veomni.distributed.moe import deepep_ace as ace

    if setting is None:
        monkeypatch.delenv("VEOMNI_MUSA_ACE_PAYLOAD_READY", raising=False)
    else:
        monkeypatch.setenv("VEOMNI_MUSA_ACE_PAYLOAD_READY", setting)
    order = []
    contiguous, as_float = torch.Tensor.contiguous, torch.Tensor.float

    def prepare_hidden(tensor, *args, **kwargs):
        order.append("contiguous")
        return contiguous(tensor, *args, **kwargs)

    def prepare_probs(tensor, *args, **kwargs):
        order.append("float")
        return as_float(tensor, *args, **kwargs)

    previous = SimpleNamespace(current_stream_wait=lambda: order.append("wait_external"))
    ready, layout = object(), object()

    def capture():
        order.append("capture_ready")
        return ready

    class Buffer:
        def get_dispatch_layout(self, *args, **kwargs):
            assert kwargs["previous_event"] is (ready if setting == "1" else previous)
            order.append("layout")
            return None, None, None, None, layout

        def dispatch(self, hidden, **kwargs):
            assert hidden.is_contiguous()
            assert kwargs["topk_weights"].dtype == torch.float32
            assert kwargs["previous_event"] is layout
            order.append("dispatch")
            return hidden, kwargs["topk_idx"], kwargs["topk_weights"], [3], object(), object()

    hidden = torch.ones(2, 3).t()
    probabilities = torch.ones(3, 1, dtype=torch.bfloat16)
    indices = torch.zeros(3, 1, dtype=torch.long)
    state = _ACEState(SimpleNamespace(size=lambda: 2), 2, 1)
    monkeypatch.setattr(state, "_get_buffer", lambda _: Buffer())
    monkeypatch.setattr(ace, "_current_stream_event", capture)
    monkeypatch.setattr(torch.Tensor, "contiguous", prepare_hidden)
    monkeypatch.setattr(torch.Tensor, "float", prepare_probs)
    state.dispatch(hidden, indices, probabilities, previous_event=previous)
    assert order == (
        ["wait_external", "contiguous", "float", "capture_ready", "layout", "dispatch"]
        if setting == "1"
        else ["layout", "contiguous", "float", "dispatch"]
    )


def test_deepep_ace_is_a_dispatcher_selection():
    config = OpsImplementationConfig(moe_dispatcher="deepep_ace")
    assert config.moe_implementation == "fused_triton"
    assert config.moe_dispatcher == "deepep_ace"
    assert config.moe_deepep_num_sms == 20
    assert config.moe_deepep_token_capacity == 8192


def test_deepep_ace_num_sms_must_be_even_and_positive():
    with pytest.raises(ValueError, match="positive even"):
        OpsImplementationConfig(moe_deepep_num_sms=3)


def test_deepep_ace_token_capacity_must_be_positive():
    with pytest.raises(ValueError, match="token_capacity must be positive"):
        OpsImplementationConfig(moe_deepep_token_capacity=0)


def test_deepep_payload_cache_scope_restored_after_excluded_layer():
    from veomni.distributed.moe import deepep_ace_checkpoint as ace

    records = []
    token = ace._CHECKPOINT_DISPATCH_CACHE.set(records)
    try:
        with ace.moe_dispatch_context(12, cache_payload=False):
            assert ace._CHECKPOINT_DISPATCH_CACHE.get() is None
        assert ace._CHECKPOINT_DISPATCH_CACHE.get() is records
        with pytest.raises(ValueError), ace.moe_dispatch_context(13, cache_payload=False):
            raise ValueError("test context cleanup")
        assert ace._CHECKPOINT_DISPATCH_CACHE.get() is records
    finally:
        ace._CHECKPOINT_DISPATCH_CACHE.reset(token)


def test_qwen_checkpoint_cache_stride_uses_decoder_layer_index(monkeypatch):
    from veomni.distributed.moe import deepep_ace_checkpoint as ace
    from veomni.models.transformers.qwen3_5.qwen3_5_musa_runtime_patch import (
        install_qwen3_5_moe_deepep_recompute_patch,
    )

    class Sparse:
        def forward(self):
            return ace._CHECKPOINT_DISPATCH_CACHE.get()

    class Decoder:
        def __init__(self, config, layer_idx):
            self.mlp = Sparse()

    module = SimpleNamespace(Qwen3_5MoeSparseMoeBlock=Sparse, Qwen3_5MoeDecoderLayer=Decoder)
    monkeypatch.setenv("VEOMNI_DEEPEP_CACHE_LAYER_STRIDE", "1")
    install_qwen3_5_moe_deepep_recompute_patch(module, cache_stride=4)
    layers = [Decoder(None, idx) for idx in range(8)]
    # A second model may choose a different stride using the same classes.
    install_qwen3_5_moe_deepep_recompute_patch(module, cache_stride=2)
    second_layers = [Decoder(None, idx) for idx in range(8)]
    # Shared-expert overlap may replace forward when constructing a later model.
    original_init = Decoder.__init__
    Sparse.forward = lambda self: ace._CHECKPOINT_DISPATCH_CACHE.get()
    install_qwen3_5_moe_deepep_recompute_patch(module, cache_stride=2)
    assert Decoder.__init__ is original_init
    records = []
    token = ace._CHECKPOINT_DISPATCH_CACHE.set(records)
    try:
        for idx, layer in enumerate(second_layers):
            assert layer.mlp.forward() is (records if idx % 2 == 0 else None)
        for idx, layer in enumerate(layers):
            assert layer.mlp.forward() is (records if idx % 4 == 0 else None)
            assert ace._CHECKPOINT_DISPATCH_CACHE.get() is records
    finally:
        ace._CHECKPOINT_DISPATCH_CACHE.reset(token)


def test_compact_cache_rejects_models_without_active_scopes():
    from veomni.distributed.moe.deepep_ace_checkpoint import validate_model_cache_scopes

    class Sparse(torch.nn.Module):
        def forward(self, value):
            return value

    model = torch.nn.Sequential(Sparse(), Sparse())
    with pytest.raises(ValueError, match="installed Qwen3.5-MoE cache scopes"):
        validate_model_cache_scopes(model)
    for idx, block in enumerate(model):
        block._veomni_cache_payload = idx % 2 == 0
    with pytest.raises(ValueError, match="installed Qwen3.5-MoE cache scopes"):
        validate_model_cache_scopes(model)
    Sparse.forward._veomni_deepep_cache_scope = True
    validate_model_cache_scopes(model)


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


@pytest.mark.parametrize("backward_order", [(0, 1), (1, 0)])
@pytest.mark.parametrize("owned_receive", [False, True])
def test_compact_checkpoint_preserves_gradients(monkeypatch, backward_order, owned_receive):
    """Exercise the public dispatcher, both backward orders, and workspace reuse."""
    from torch.utils.checkpoint import checkpoint

    from veomni.distributed.moe import deepep_ace as ace
    from veomni.distributed.moe import deepep_ace_checkpoint as cache

    class Event:
        def current_stream_wait(self):
            pass

    class Buffer:
        calls = 0
        workspace = torch.zeros(3, 4)

        def get_dispatch_layout(self, *args, **kwargs):
            return None, None, None, None, Event()

        def dispatch(self, hidden_states, **kwargs):
            self.calls += 1
            self.workspace.copy_(hidden_states)
            received = self.workspace.clone() if owned_receive else self.workspace
            return received, kwargs["topk_idx"].clone(), kwargs["topk_weights"].clone(), [3, 3], (), Event()

    class Expert:
        @staticmethod
        def apply(compact, counts, weight):
            return (compact @ weight).sin()

    buffer = Buffer()
    group = SimpleNamespace(size=lambda: 2)
    monkeypatch.setattr(ace, "get_parallel_state", lambda: SimpleNamespace(ep_enabled=True, ep_group=group))
    monkeypatch.setattr(ace, "_supported_ep_classes", lambda: (Expert,))
    monkeypatch.setattr(ace, "_current_stream_event", Event)
    monkeypatch.setattr(ace._ACEState, "_get_buffer", lambda *args: buffer)
    monkeypatch.setattr(ace._ACEState, "reverse_combine", lambda self, x, p: (x, p))
    monkeypatch.setattr(ace._ACEState, "combine", lambda self, x: x.clone())
    monkeypatch.setattr(ace._ACEState, "reverse_dispatch", lambda self, x: x)
    monkeypatch.setattr(ace, "_ace_te_supported", lambda *args: False)
    config = SimpleNamespace(moe_dispatcher="deepep_ace", moe_deepep_recompute_cache="none")
    monkeypatch.setattr(cache, "get_ops_config", lambda: config)
    idx = torch.tensor([[0, 1], [1, 0], [0, 1]])

    def block(x, probs, weight):
        return ace.dispatch_to_ep_class_deepep_ace(Expert, 4, probs, idx, x.sin(), weight)

    torch.manual_seed(23)
    inputs = [tuple(torch.randn(*s) for s in ((3, 4), (3, 2), (4, 4))) for _ in range(2)]

    def run(mode):
        config.moe_deepep_recompute_cache = mode
        buffer.calls = 0
        xs = [tuple(t.clone().requires_grad_() for t in sample) for sample in inputs]
        ys = [checkpoint(block, *x, use_reentrant=False, context_fn=cache.deepep_recompute_context_fn) for x in xs]
        buffer.workspace.fill_(999)
        for i in backward_order:
            ys[i].square().sum().backward()
        assert cache._CHECKPOINT_DISPATCH_CACHE.get() is None
        assert not cache._IN_CHECKPOINT_RECOMPUTE.get()
        return ys, [t.grad for sample in xs for t in sample], buffer.calls

    baseline, baseline_grads, baseline_calls = run("none")
    cached, cached_grads, cached_calls = run("compact")
    assert baseline_calls == 4 and cached_calls == 2
    for a, b in zip(baseline + baseline_grads, cached + cached_grads):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
