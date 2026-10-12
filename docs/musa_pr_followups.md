# Further MUSA PR15/16/17 experiments

These independent experiments extend the existing default-off PR15/16/17
implementations. Every new flag below defaults to `0`. They do not change the
training launcher or generated model files. Use the same flags on all ranks;
check for unconditional exports in the launcher before an A/B comparison.

| Additional environment flag (`=1`) | Required original flag | Work removed or combined |
| --- | --- | --- |
| `VEOMNI_MUSA_VISION_INTERPOLATION_CACHE` | `VEOMNI_MUSA_VISION_EMBEDDING_CSR` | Cache the grid-dependent interpolation indices and coefficients as well as the existing CSR grouping. |
| `VEOMNI_MUSA_QK_JOINT_FORWARD` | `VEOMNI_MUSA_QK_PREENTRY` | Produce normalized, repeated Q and K in one kernel launch. |
| `VEOMNI_MUSA_QK_FUSED_BACKWARD` | `VEOMNI_MUSA_QK_PREENTRY` | Fuse normalization backward and repeated-head reduction, avoiding a global 32-head gradient intermediate. |
| `VEOMNI_MUSA_SUPERVISED_CE_METADATA` | `VEOMNI_MUSA_SUPERVISED_CE` | Select supervised causal indices in the CPU collator, avoiding dynamic device selection in the loss. |
| `VEOMNI_MUSA_ACE_COMPACT_STATE_REUSE` | Native TE; `VEOMNI_MUSA_ACE_TE_SLOT_PROBABILITY` for the forward-only primitive | Avoid redundant inner-autograd state and reuse the forward row map within the same outer dispatch call. |
| `VEOMNI_MUSA_ACE_TE_FUSED_COUNTS` | `VEOMNI_MUSA_ACE_TE_LONG_COUNTS` | Reduce a boolean routing mask with bounded int32 partials and write the final int64 counts directly. |
| `VEOMNI_MUSA_ROUTER_FP32_PAYLOAD` | Qualified `fused_musa` + EP + `deepep_ace` | Preserve BF16 rounding while producing the FP32 communication payload in one kernel. |
| `VEOMNI_MUSA_FSDP_PACKED_PADDING` | `VEOMNI_MUSA_FSDP_SHARD_PADDING` | Align physical rank rows during existing FSDP packing, avoiding the old wrapper's extra full-buffer pack/unpack copies. |

## Numerical and lifetime contracts

The interpolation cache stores immutable grid metadata, never embedding
weights or outputs. It uses per-module weak references, a bounded 128-entry
LRU, ready events and stream recording. An entry created during inference is
also valid for later training. Weight updates still take effect immediately.

The joint QK forward retains the native backward by default. The separate
fused-backward experiment is evaluated under BF16 tolerance,
**not bitwise equivalence**. In the seeded long-sequence comparison,
2 of 16,777,216 gradient elements differed, with maximum absolute error
0.00048828125. Tests use `rtol=0.02` and `atol=5e-4`, scaling the absolute
tolerance with the upstream gradient in tiny/zero-gradient cases. Both flags
can be selected together. Existing source guards and native autotune choices
remain; cold or unsupported keys fall back to the original implementation.

Supervised-CE metadata is built after packing-boundary masks and padding, only
when SP is disabled, classification is disabled and the model metadata hook
is present. It is carried under `multimodal_metadata['_veomni_supervised_ce']`.
Any later code that changes labels must discard that metadata. Explicit
`shift_labels` and unsupported metadata use the original selection path.
Projection, loss denominator and adjoint arithmetic are unchanged.

ACE state reuse is limited to a single dispatch. It never adds a full row map
or consumed communication event to the persistent checkpoint cache. The normal
grad-enabled compact operation retains its original saved-state behavior.
Routing replay validation and reverse communication remain unchanged.

Fused counts accept only contiguous MUSA boolean `[N,32]` masks with
`0 < N <= 65536`. Other inputs use the established reduction. Token lengths
are runtime arguments, preventing a new compiled kernel for every received
token count. This path does not substitute host receive counts for device
routing counts or add a metadata transfer.

The router payload preserves the original FP32-to-BF16-to-FP32 rounding in
both forward and backward. It remains an ordinary differentiable Tensor, so
FSDP's post-backward hook still sees both hidden states and routing weights.
Only marked Qwen experts receive a copied mixed-precision policy with input
casting disabled; parameter, reduction and output dtype policies stay intact.
The adapter retains the old hidden-state cast and preserves FP32 routing only
for the qualified fused-MUSA, BF16, EP and ACE path. Replay and unsupported
producer inputs retain the original rounding/cast path. An earlier opaque
carrier prototype was rejected because it hid the routing branch from the
FSDP hook. Its full training run stalled in a driver kernel launch; the causal
connection between that stall and the hook change was not established.

## Packed collective alignment

The intended benefit of padding is 16-byte alignment of each rank's
communication slice. It can help an unaligned collective use an efficient
transfer path. Already aligned rows need no padding and offer no such saving.
The old wrapper makes additional full-size staging copies, which can outweigh
the communication benefit. The new experiment writes the aligned layout in
the existing FSDP packing operation instead, with only a tiny zero tail.
Logical parameter and gradient sizes are unchanged.

This adapter pins the installed FSDP collective source and checks every
function alias before replacing bindings, including torch_musa's copy-out and
reduce aliases. Only explicitly marked communication objects use aligned
packing. It requires an eight-rank MUSA group, BF16/FP32, a rank slice of at
least one million elements, `COMM_TYPE=0` and effective `OVERLAP_LEVEL=0`.
Reduce-scatter with HSDP, a partial reduction output, an all-reduce hook or an
output-pool copy retains the original packing. Unsupported source versions
fail before installation.

## Measurement

The reference for total PR15/16/17 benefit is **all their experimental flags
disabled**, while retaining the launcher's pre-existing optimizations. The
earlier matched all-off measurement was 2978.12 ms. Enabling only supervised
CE, only vision CSR, or only QK preentry measured 2916.08, 2963.92, and
2961.88 ms respectively: observed reductions of 2.08%, 0.48%, and 0.55%.
See [the original comparison](musa_qwen35_compute.md#observations-on-2026-10-12).

The follow-up table below instead measures incremental changes over the six
previously integrated PR15/16 flags enabled, with padding disabled. Its
2889.67 ms control is not the all-off baseline. Do not add percentages from
the two comparisons or interpret their different-time means as a matched
total-speedup measurement. Each new option is measured independently.
Padding also has an old-wrapper control. Use matching inputs,
warmup, all-rank synchronized unprofiled `VLMTrainer.train_step` timing and
runtime hit counters. This timer excludes the measurement barrier and
post-step hashing/logging, so it is not the launcher's complete iteration
throughput. Concurrent workloads were explicitly allowed for the local tests;
their ownership samples are retained and shared-resource results must not be
presented as exclusive-device speedups.

Local per-run source manifests, flags, all-rank inputs/metrics, test logs,
rejected prototypes and measurements are in
`traces/pr_followups_20261012/`. Operator checks include real collator output,
the TE checkpoint path in both backward orders, the router/FSDP/experts-adapter
boundary, and eight-rank synthetic FSDP parameter/gradient parity. These checks
do not establish long-run convergence.

## Follow-up observations (2026-10-12)

Each candidate completed two valid 20-step runs; three baseline runs and two
old-padding controls also completed. Inputs matched on every rank and step.
The table uses steps 6–20, taking the maximum synchronized training-function
time across eight ranks for each step. The baseline mean is 2889.67 ms
(individual runs: 2882.49, 2909.32, 2877.19 ms). Positive reduction means faster.

| Arm | Run means, ms | Combined mean, ms | Observed latency reduction |
| --- | ---: | ---: | ---: |
| Baseline: original six flags on | 2882.49 / 2909.32 / 2877.19 | 2889.67 | +0.00% |
| CSR interpolation metadata cache | 2858.65 / 2875.92 | 2867.29 | +0.77% |
| ACE temporary-state reuse | 2848.68 / 2927.52 | 2888.10 | +0.05% |
| QK backward fusion (BF16 tolerance) | 2858.87 / 2877.84 | 2868.35 | +0.74% |
| Joint QK forward | 2914.71 / 2882.80 | 2898.76 | -0.31% |
| CPU CE metadata | 2919.26 / 3088.03 | 3003.64 | -3.94% |
| Fused counts | 2946.60 / 2873.22 | 2909.91 | -0.70% |
| Ordinary FP32 router payload | 2888.53 / 3163.59 | 3026.06 | -4.72% |
| Old padding wrapper | 2893.46 / 2926.20 | 2909.83 | -0.70% |
| Packing-integrated padding | 2870.00 / 2891.64 | 2880.82 | +0.31% |

Concurrent compilation and attention tests were present. CE run 2 and payload
run 2 had substantial external activity; their slower means are not isolated
algorithm regressions. The three baseline run means themselves span about 1.1%.
These shared-resource observations do not establish stable improvements, and
none of the new flags is enabled by default.

Packing-integrated padding was 1.00% faster than the old wrapper on the combined
means, but only 0.31% faster than padding disabled. Both implementations hit
54 all-gathers and 27 reduce-scatters per rank per step, corresponding to the
27 visual blocks with checkpoint recomputation. The only affected rank slices
were BF16 AG 1904938→1904944 and FP32 RS 1904938→1904940. The new path recorded
no old-wrapper padding calls. Other candidates also recorded their expected
runtime hits; detailed counts and per-run interference are in the local report.

Validation: 38 follow-up contracts, 47 ACE regression tests, eight-rank synthetic
FSDP padding checks, and a two-rank synthetic expert test with nested FSDP,
checkpointing and three FusedAdamW updates passed. The router test checked
loss, input/parameter gradients and updated parameters exactly and asserted
fast-path hits; it does not substitute for the real ACE training runs.
All 21 completed runs had finite loss/gradient norms and matching input hashes.
Baseline repeats differed in loss by up to 0.0108034; the maximum across all
arms was 0.0172250. Full training trajectories are not bitwise equivalent.

Rejected prototypes and interrupted attempts are retained separately: a
length-specialized counts kernel caused excessive JIT compilation; an early
router adapter reintroduced a BF16 cast; the opaque payload changed the FSDP
hook boundary and its full run stalled in a driver launch with causation
unresolved; a shared-device QK-forward repeat hit GPU3 OOM and was rerun.
They are excluded from the valid timing table, with source and failure logs
retained in `traces/pr_followups_20261012/`.

## Recommended candidate configuration

For the local Qwen3.5 eight-rank launcher, the performance-oriented candidate
enables supervised CE, vision CSR with interpolation metadata caching, QK
preentry with fused backward, and packing-integrated padding. It leaves the
other PR16 and follow-up experiments disabled. QK backward uses the BF16
tolerance described above; both padding switches and the no-overlap/default
communication settings are required.

This selection is based on individual observations. The complete combination
has not been directly compared with all flags off, so it is not a measured
optimum and has no established combined speedup. Library defaults remain off.
The existing local launcher is not part of this source change; it must allow
these environment values to reach the workers without overriding them.

```bash
env \
  PROFILE_ENABLE=0 \
  TORCH_MUSA_FSDP2_COMM_TYPE=0 \
  TORCH_MUSA_FSDP2_OVERLAP_LEVEL=0 \
  VEOMNI_MUSA_SUPERVISED_CE=1 \
  VEOMNI_MUSA_VISION_EMBEDDING_CSR=1 \
  VEOMNI_MUSA_VISION_INTERPOLATION_CACHE=1 \
  VEOMNI_MUSA_QK_PREENTRY=1 \
  VEOMNI_MUSA_QK_FUSED_BACKWARD=1 \
  VEOMNI_MUSA_FSDP_SHARD_PADDING=1 \
  VEOMNI_MUSA_FSDP_PACKED_PADDING=1 \
  VEOMNI_MUSA_QK_JOINT_FORWARD=0 \
  VEOMNI_MUSA_SUPERVISED_CE_METADATA=0 \
  VEOMNI_MUSA_ACE_TE_SLOT_PROBABILITY=0 \
  VEOMNI_MUSA_ACE_TE_LONG_COUNTS=0 \
  VEOMNI_MUSA_ACE_PAYLOAD_READY=0 \
  VEOMNI_MUSA_ACE_COMPACT_STATE_REUSE=0 \
  VEOMNI_MUSA_ACE_TE_FUSED_COUNTS=0 \
  VEOMNI_MUSA_ROUTER_FP32_PAYLOAD=0 \
  bash run_qwen35_35b_a3b_8card_train.sh
```
