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
`moe_shared_expert_overlap=true` only for the side-stream scheduling
experiment. It is not implicitly enabled by selecting this backend, and a
trace must confirm useful concurrency before treating it as a speedup.
