# Qwen3.5 MUSA compute experiments

Three independent, default-off experiments are adapted from
[PR #15](https://github.com/arcing-mt/VeOmni/pull/15). They target the existing
Qwen3.5-MoE MUSA training path. This source change does not include launcher
changes or edits to generated model sources. Set a flag to `1` before starting
training to select that experiment;
unset flags and `0` retain the existing implementation.

| Environment flag | Change | Required runtime path |
| --- | --- | --- |
| `VEOMNI_MUSA_SUPERVISED_CE` | Project only causal positions with supervised targets into vocabulary logits | MUSA `chunk_loss`, SP disabled, partially ignored labels |
| `VEOMNI_MUSA_VISION_EMBEDDING_CSR` | Reuse sorted interpolation indices and accumulate positional embedding gradients with a CSR kernel | Qualified Qwen3.5 vision interpolation, BF16 embedding `[2304,1152]`, more than 3072 indices |
| `VEOMNI_MUSA_QK_PREENTRY` | Fuse query/key contiguous copy, head repetition and L2 normalization | Qualified Qwen3.5 GDN and `musa_tilelang`, BF16 packed batch 1, 16 key heads repeated to 32 value heads |

These are experimental switches within existing implementations, not new
values of `model.ops_implementation.*`. Existing model/backend selection is
still required. To select only supervised CE, the environment reaching the
training process must contain:

```bash
export VEOMNI_MUSA_SUPERVISED_CE=1
export VEOMNI_MUSA_VISION_EMBEDDING_CSR=0
export VEOMNI_MUSA_QK_PREENTRY=0
```

A launcher can override these values with its own assignments. Check such
assignments before an independent comparison; setting an environment value
outside a script cannot override an unconditional export inside that script.

## Supervised-token loss

Ignored labels do not contribute to the loss but the original implementation
still projects their hidden states into the full vocabulary. After the causal
shift, the experiment gathers only positions whose labels differ from
`ignore_index`. It then calls the existing chunked projection/loss and adjoint
without shifting the compacted labels again. Transformer layers still process
the complete sequence: prompt and image tokens remain available as context.

The benefit depends on the supervised fraction and the projection's share of
step time. Compaction adds a dynamic `nonzero` and gathering cost, and changes
chunk boundaries and BF16 weight-gradient summation order. It does not promise
bitwise gradient equivalence. All-supervised, all-ignored and SP inputs use the
native path. Explicit `shift_labels` are treated as already aligned when the
experiment is enabled; the disabled API behavior is unchanged.

## Vision embedding backward

At the guarded interpolation site, index values depend only on the image grid.
The first encounter with a grid builds stable sorted slots and row offsets;
subsequent calls reuse them. Forward still uses native embedding. Backward
sums each embedding row's contributors in FP32 groups of at most ten entries,
then casts to BF16, matching the qualified native build's tested arithmetic.

The first cache miss includes D2H, CPU sorting and H2D transfers. The cache is
per embedding module, keyed by `(height, width, grid_size, device)`, and limited
to 128 grids with LRU eviction. Events and stream recording protect metadata
reuse. Small index sets, embedding hooks, child FSDP, DTensor weights and
unsupported layouts use the native module; higher-order backward uses native
embedding backward. This helper is not a general cache for arbitrary indices.

This is independent of `vision_patch_embed_implementation=linear`. The existing
linear path projects pixel patches into visual features, replacing the special
Conv3d with GEMM. CSR instead handles gradients of the positional lookup table
whose interpolated vectors are added to those features. Both paths can run
together; the measurements below retain the existing linear patch projection.
Only index grouping is cached: position-table weights and gradients continue
to be computed and updated each step.

## GDN query/key preentry

The fused forward reads the strided 16-head query/key input directly and emits
normalized 32-head output, avoiding intermediate copies and separate launches.
Backward retains the native 32-head BF16 L2 normalization derivative and then
sums each repeated head pair, preserving the native rounding order.

The adapter checks exact hashes of the generated model, loaded GDN forward,
MUSA backend, installed FLA normalization and autotune-cache implementation.
An explicitly selected experiment raises on unsupported source versions.
Supported sources with unsupported inputs, cache/inference/SP modes or a cold
native autotune key fall back to native work. Cold keys first run the original
autotuner; the experiment reads its choice without changing it. Higher-order
gradients are unsupported by this normalization path.

The local adapter preserves the existing host `cu_seqlens` metadata scope and
supports the local undecorated GDN forward. Device contexts around direct
Triton calls support tensors on a non-current MUSA device.

## Validation and measurement

`tests/ops/test_musa_qwen35_compute.py` covers causal/explicitly shifted CE,
ignored-label boundaries, SP/CPU fallbacks, MUSA BF16 CE gradients, bitwise
QK and CSR forward/backward parity, cache reuse, another stream, non-current
devices, and complete GDN layer output/parameter gradients. CSR also checks
the higher-order native fallback.

For performance, compare each flag separately against all three flags off,
using the same original launcher, inputs, seed, runtime and existing optimization
settings. Exclude warmup and use exclusive devices for stable speedup claims.
If concurrent workloads are explicitly allowed, record their presence and label
the result as shared-resource observations. Measure synchronized wall time on
every rank without profiling and report the slowest rank per step; do not infer
step improvement from isolated kernel timings.
The modules expose `stats()` to distinguish successful installation from real
runtime hits, fallbacks and cache misses.

### Observations on 2026-10-12

The existing Qwen3.5-35B-A3B image launcher ran on eight MTT S5000 devices with
torch/torch_musa 2.9.1, transformers 5.9.0, FSDP2=8, EP=8, SP=1, micro batch 2
and global batch 16. Existing linear patch projection, ACE direct writes and
compact checkpoint cache stride 2 stayed enabled. PR16's three experimental
flags and PR17 shard padding stayed off in every arm.

Each arm ran twice from the same initial model, in the order baseline, CE, CSR,
QK, QK, CSR, CE, baseline. Each run completed 20 steps; the table averages steps
6-20 from both runs, taking the maximum time across eight ranks for each step.
All per-rank, per-step input hashes matched. Foreign GPU contexts were sampled
in every run, including intermittent external compilation and attention tests.
Concurrent testing was explicitly authorized; these are indicative observations,
not evidence of stable gains on exclusive devices.

| Arm | First run ms | Second run ms | Combined mean ms | Latency reduction |
| --- | ---: | ---: | ---: | ---: |
| All three off | 2980.88 | 2975.37 | 2978.12 | — |
| Supervised CE only | 2907.72 | 2924.43 | 2916.08 | 2.08% |
| Vision CSR only | 2961.69 | 2966.14 | 2963.92 | 0.48% |
| QK preentry only | 2951.67 | 2972.08 | 2961.88 | 0.55% |

These times measure the existing `VLMTrainer.train_step` with a host wall clock
and device synchronization, without a profiler. The timer includes batch
retrieval, forward/backward, communication, gradient clipping, optimizer and
step callbacks. It excludes outer iterator creation, the measurement's pre-step
rank barrier, and post-step input hashing/logging. These additional operations
also change the cadence between steps. It is therefore a training-function
measurement, not the launcher's complete iteration throughput or a GPU kernel
sum. For example, baseline1 step 6 measured 2.962 seconds on rank 0 while tqdm
showed 3.11 s/it at its update and 3.29 s/it when closing the epoch; the final
refresh additionally included measurement bookkeeping.

Runtime activation, per 20-step run across all eight ranks:

- CE: 160 calls, no fallback; vocabulary-projection positions decreased from
  1,294,288 to 523,769 (40.47% retained). Full transformer sequences remained.
- CSR: 2,321 optimized backward calls, 2,133 cache hits, 188 grid-cache builds
  and 355 native fallback calls. Cache construction costs were not removed
  from measured steps.
- QK: 19,192 fused forwards and 9,592 corresponding backwards, with eight cold
  native-configuration fallbacks. Each steady step used 120 fused forwards
  (including checkpoint recomputation) and 60 backwards per rank. Backward
  still uses native L2 normalization and repeated-head summation.

CE showed the largest observed reduction. CSR and QK changed only small parts
of the training function, with sub-0.6% observed differences that do not establish
stable improvements under these shared-resource conditions. Peak allocated
memory stayed approximately 71.98-71.99 GiB across arms.

The 15 new correctness tests and one existing chunk-loss regression passed,
as did static review and `make quality`. All eight training runs completed with
finite losses and gradient norms. Full training trajectories were not bitwise
equal: the maximum loss difference against baseline1 was 0.00706 for baseline2,
0.00507 for CE, 0.01051 for CSR and 0.01631 for QK. Operator parity tests do not
establish long-run convergence or full-model gradient identity.

Local measurement artifacts are retained under `traces/pr15_ab_20261012/`:
`report.md`, `summary.json`, `source_manifest.json`, operator test logs, and each
run's per-rank metrics, input hashes and process-ownership records. These local
artifacts and the training launcher are not part of the source change.
