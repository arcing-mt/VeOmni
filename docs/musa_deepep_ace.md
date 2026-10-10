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

On MUSA, routing validation reduces a contiguous `[experts, tokens]` copy of
the multi-hot map. The tested torch_musa 2.9.1 runtime can undercount a tall
`[tokens, experts]` map when reducing dimension 0 (for example `[32640, 32]`),
incorrectly sending valid routing through the slower PyTorch fallback. The
contiguous reduction preserves exact integer counts. Duplicate-slot and
count validation remains available through the explicit validation path; this
requires no extra option.

The production TE path now trusts DeepEP's host-side per-expert receive counts
for `num_out`, while the corrected contiguous GPU reduction supplies the
grouped-GEMM cumulative counts. It does not convert the GPU mask counts to
Python or copy the host counts back to the device on every MoE call. Full duplicate-slot and count
validation remains available through the private `validate_routing=True` test
and diagnostic path; native Qwen top-k produces distinct expert indices, and
DeepEP has already generated the receive counts from the same dispatch. A
custom router-replay implementation that can emit duplicate expert indices must
use the validation path before enabling TE compaction. The Qwen3.5 runtime
automatically enables that validation context while a router-replay manager is
active, so replay pays the old protection cost without weakening the normal
native-top-k path.

## Selective checkpoint caching

Select the policy through `model.ops_implementation` (CLI or YAML):

| Parameter | Default | Meaning |
| --- | --- | --- |
| `moe_deepep_recompute_cache` | `none` | `none`: normal recomputation; `compact`: retain the compacted expert input to skip forward dispatch and compaction in selected layers. |
| `moe_deepep_cache_layer_stride` | `2` | Positive integer. With `compact`, cache zero-based decoder indices divisible by the stride. `2` selects layers 0, 2, 4, ...; `1` selects every layer. Ignored in `none` mode. |

```bash
--model.ops_implementation.moe_deepep_recompute_cache compact \
--model.ops_implementation.moe_deepep_cache_layer_stride 2
```

```yaml
model:
  ops_implementation:
    moe_dispatcher: deepep_ace
    moe_deepep_recompute_cache: compact
    moe_deepep_cache_layer_stride: 2
train:
  gradient_checkpointing:
    enable: true
    enable_reentrant: false
```

Set `moe_deepep_recompute_cache=none` to restore the normal path; no other
cache setting needs changing. All EP ranks must use the same arguments.
The explicit configuration is logged and saved in `veomni_cli.yaml`.

The compact policy currently supports the Qwen3.5-MoE MUSA runtime path with
enabled non-reentrant gradient checkpointing. Checkpoint debug mode and
`torch.compile` are rejected. It does not disable global checkpointing, expert
GEMMs, shared-expert recomputation, combine, or either gradient collective.
Invalid modes, non-positive/non-integral strides, and non-ACE dispatchers are
rejected before training. The selected model must expose the Qwen3.5-MoE
runtime cache scopes.

Each checkpoint invocation owns its compact activation, routing plan, and
completion event. The original received payload is released; large int64 TE
row maps are rebuilt from the small bool routing map. A single autograd
Function owns dispatch and compaction on both passes, preserving saved-tensor
metadata order and routing-weight gradients. Records are consumed during
recomputation and released on context exit, including early-stop. Deterministic
checkpoint routing and a single backward per invocation are required.

Caching every layer is not necessarily faster: the measured partial-layer
policy avoids the regression seen with full-layer caching. Historical timings,
traces, and validation limits are recorded in
`traces/deepep_compact_boundary_20261008/README.md`; they are not a performance
claim for arbitrary launcher/data changes.

### Migration from the experimental switches

The three experimental booleans (`moe_deepep_reuse_layout_on_recompute`,
`moe_deepep_cache_dispatch_on_recompute`, and
`moe_deepep_reuse_dispatch_handle_on_recompute`) have been removed along with
the layout/handle/global-stack and received-payload-cache implementations.
Old CLI flags are rejected; old YAML keys produce an explicit migration error.

The earlier `VEOMNI_DEEPEP_*` cache/recompute environment switches no longer
control any active code, including the compact-dispatch and layer-stride
switches. Remove them and set the two arguments above. Retired experiment
artifacts under `traces/` retain their original names for reproducibility.
The unrelated `VEOMNI_ACE_TE` kernel-fallback override is unchanged.

### Direct writes into registered combine windows

Set `model.ops_implementation.moe_deepep_combine_direct_write: true` to enable
the direct-window producer in the normal ACE dispatcher. The config default is
`false`; the local `run_qwen35_35b_a3b_8card_train.sh` explicitly enables it:

```bash
# Direct-window path, 1 step per epoch x 20 epochs.
bash run_qwen35_35b_a3b_8card_train.sh
```

The local launcher does not forward trailing arguments. To run its baseline
producer/combine path, change its existing
`--model.ops_implementation.moe_deepep_combine_direct_write true` argument to
`false`, keeping the other settings unchanged.

For entrypoints invoked through `train.sh`, pass these training arguments
directly (use `false` to disable direct writes):

```bash
--model.ops_implementation.moe_dispatcher deepep_ace \
--model.ops_implementation.moe_deepep_combine_direct_write true
```

```yaml
model:
  ops_implementation:
    moe_dispatcher: deepep_ace
    moe_deepep_combine_direct_write: true
```

The resolved argument is logged and saved in `veomni_cli.yaml`. It must agree
on every EP rank and remain fixed for the lifetime of the training process.
Enabling it with another dispatcher is rejected before training. No experimental
entrypoint or patched DeepEP/MCCL library is needed.

The experimental `VEOMNI_ACE_COMBINE_DIRECT_WRITE` and
`VEOMNI_ACE_COMBINE_STAGING` environment variables no longer control active
code. Remove both from launch commands. Staging was a separate experiment
that first copied ordinary producer outputs into a registered window;
`VEOMNI_ACE_COMBINE_STAGING=0` merely selected its default disabled state.
That copy-based branch has been removed. Direct writes do not depend on it.

The enabled path uses **two** registered DeepEP windows and requires MUSA BF16,
native TransformerEngine compaction, and `Buffer.get_ace_combine_buffer`.
Unsupported payloads or routing layouts raise an error instead of silently
falling back. All-to-all dispatchers are unaffected. TE unpermute writes the
weighted expert sums directly into the leased window; backward TE unpermute and
the routing-gradient gather also write there. GEMM, routing probabilities and
the FP32 accumulation order are preserved. This removes producer-to-window
staging copies; it does not remove CE peer transfers or all model copies.

The fixed two-slot pool waits for the prior completion event before each write
and retains the lease through submission. Checkpoint early-stop releases a
producer event without adding a collective. Empty receivers retain a non-null
view of the selected slot. Window contents are not saved activations, and
module-level lease objects release their references on exit, including when
Python cyclic GC is disabled. The existing compact checkpoint cache and layer
stride remain supported.

The validated experimental implementation completed 1 step x 20 runs at
3.039--3.054 s/step versus a 3.052 s/step baseline (steps 11--20, rank maximum).
Matched rank-0 traces showed SendRecv kernels decreasing from 360 to zero over
three profiled steps. These measurements do **not** establish an end-to-end
speedup. Small fixed-input output/gradient parity passed; full-model training
loss/gradient trajectories are not bit-identical, and long-run convergence was
not validated. The second registered window increases workspace allocation.
