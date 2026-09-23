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

## Expert compaction on TransformerEngine kernels

Compacting the received rows into the expert-major layout the grouped GEMM reads,
and summing them back afterwards, runs on TransformerEngine kernels:

| stage | kernel |
|---|---|
| row ids | `transformer_engine.pytorch.triton.permutation.make_row_id_map` |
| permute | `tex.moe_permute_mask` (activations and routing weights in one kernel) |
| un-permute | `tex.moe_unpermute_mask` (FP32 accumulator, fixed visit order) |
| permute backward | `tex.moe_unpermute_mask`, replacing the BF16 atomic `index_add_` |
| un-permute backward | `tex.moe_permute_mask`, replacing `index_select` |

The row order is the one the PyTorch path builds with `nonzero` + stable
`argsort` (expert ascending, token ascending inside an expert), so the grouped
GEMM sees identical segments and the permute stage is bit-identical. The
un-permute and both backward paths round differently by design: the un-permute
matches bit-for-bit over the measured shapes because both accumulate in FP32, and
the permute backward moves from BF16 atomics to an FP32 sum. What changes:

* The Triton row map costs 0.22 ms per layer instead of 1.00 ms.
* The permute backward no longer accumulates duplicate rows with BF16 atomics:
  against an FP64 reference that path measured 4.0e-3 and was not run-to-run
  reproducible, where the FP32 kernel measures 6.9e-7 and is.
* The un-permute backward is bit-identical to the `index_select` it replaces and
  31% faster (16-byte vector moves instead of int64-indexed row copies).

End to end on Qwen3.5-35B-A3B (8x MTT S5000, EP8, FSDP2, `chunk_loss`) the TE
path measures 3.19 s/step against 3.42 s/step for the FP32 `index_add_` fallback:
it keeps the FP32 accumulator's accuracy without paying for it.

`VEOMNI_ACE_TE=0` forces the PyTorch path. The MUSA kernels are 16-bit only, so
the TE path needs a contiguous MUSA payload of BF16/FP16 with a hidden size that
is 16-byte-vector aligned, FP32 routing weights and a local-expert count divisible
by four; otherwise the whole compaction falls back, including on every CPU test.
The two paths are driven by different maps (expert-major against top-k slot
order), so they must not be mixed halfway.
