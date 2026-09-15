# MUSA DeepEP-ACE MoE backend

VeOmni exposes the MUSA DeepEP-ACE path as an opt-in MoE implementation:

```yaml
model:
  ops_implementation:
    moe_implementation: fused_musa
    moe_dispatcher: deepep_ace
```

The backend requires a MUSA DeepEP build with ACE support. It caches
`Buffer(use_ace=True, train_mode=True)` per process-group/shape/resource tuple,
uses the communication stream for dispatch/combine, and keeps a per-invocation
communication handle through the custom backward path. The default
`fused_triton`/MCCL all-to-all path is unchanged.

This integration provides ACE dispatch/combine and explicit asynchronous
completion while leaving the expert compute selection in `moe_implementation`.
It does not yet pipeline multiple token chunks with sparse expert GEMM; that is
a separate optimization. The first validation should therefore compare
forward/backward outputs on fixed input and inspect the dispatch, grouped-GEMM,
and combine intervals in a multi-card MUSA trace.

The current Qwen3.5 model also has a shared expert. Scheduling that independent
shared expert before ACE dispatch is a later overlap optimization and is not
implicitly enabled by selecting this backend.
