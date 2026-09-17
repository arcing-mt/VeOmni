# MUSA DeepEP-ACE MoE backend

VeOmni exposes the MUSA DeepEP-ACE path as an opt-in MoE implementation:

```yaml
model:
  ops_implementation:
    moe_implementation: fused_musa
    moe_dispatcher: deepep_ace
```

The backend requires a MUSA DeepEP build with ACE support. It caches
`Buffer(use_ace=True)` (and passes `train_mode=True` only when the installed
wheel exposes that constructor argument) per process-group/shape/resource tuple,
uses the communication stream for dispatch/combine, and keeps a per-invocation
communication handle through the custom backward path. The default
`fused_triton`/MCCL all-to-all path is unchanged.

This integration provides ACE dispatch/combine and explicit asynchronous
completion while leaving the expert compute selection in `moe_implementation`.
The ACE workspace has a fixed `moe_deepep_token_capacity` and all EP ranks must
use the same value. The first validation should therefore compare
forward/backward outputs on fixed input and inspect the dispatch, grouped-GEMM,
and combine intervals in a multi-card MUSA trace.

The current Qwen3.5 model also has a shared expert. Set
`moe_shared_expert_overlap=true` to issue it inside the ACE dispatch window:
the dispatcher submits the dispatch and *then* runs the shared expert, so the
shared expert executes while the payload is in flight. It runs on the compute
stream and therefore does not overlap the grouped GEMM or the combine, and it is
not implicitly enabled by selecting this backend.

Do not give the shared expert a stream of its own. Measured on Qwen3.5-35B-A3B
(8x MTT S5000, EP8, FSDP2, `chunk_loss`), a dedicated high-priority side stream
cost ~0.6 s/step relative to `moe_shared_expert_overlap=false` and *lowered* the
compute/comm overlap from 367 ms to 267 ms per step: every layer paid a
`record_event`/`wait_event` pair and a stream switch for each of its kernels,
the autograd graph was split across two streams, and the compute stream drained
before the side stream filled, so the GPU starved between layers (idle rose from
747 ms to 1242 ms). Confirm any change here with
`scripts/profile/moe_overlap_trace_ab.py`, which reports the compute/comm
overlap, cross-stream overlap and idle-gap histogram for two traces.
