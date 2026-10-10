---
orphan: true
---

# Opt-in MUSA DeepEP/ACE TE compaction

Normal DeepEP/ACE dispatch now selects one matched permute/unpermute implementation directly. No historical-directory imports, AST rewriting or qualification JSON paths are required. Default compaction remains native.

| Flag (`=1`) | Purpose |
| --- | --- |
| `VEOMNI_MUSA_ACE_TE` | TE native expert-major hidden gather/unpermute, fixed-order FP32 accumulation |
| `VEOMNI_MUSA_ACE_TE_LONG_COUNTS` | Long input column sums for qualified E32 routing shapes with incorrect Bool reductions |
| `VEOMNI_MUSA_ACE_TE_SLOT_PROBABILITY` | Direct top-k probability copies/gradients around unchanged TE hidden arithmetic |
| `VEOMNI_MUSA_ACE_NATIVE_WARMUP` | One-shot synthetic fallback initialization inside the first real compaction; requires `VEOMNI_MUSA_DEEPEP_COUNTING_SORT=1` |

All flags default to `0`; counts/slot require the TE path to be selected. Native warmup is startup support, not a steady-state speedup. All index/count/duplicate checks remain before slot kernels. Empty/unsupported payloads or unavailable TE use native compaction. TE import restores its modified global Torch APIs before other training continues. Already-imported TE or a different Torch/MUSA build falls back with a warning.

Supported target: torch2.11.0.post2 / torch_musa2.11.0.post2+395c00c; MUSA TE from [MR156](https://sh-code.mthreads.com/ai/TransformerEngine/-/merge_requests/156), target `arcing/mudnn3.4`. Counts cast is restricted to captured N ranges (24577–28608 or 28673–32704), E32/H2048/topk8; probability copy to BF16 H2048/E32/topk8.

The payload-ready fix is included: communication events cover contiguous hidden copies and FP32 probability casts, including shared-expert overlap. TE metadata stays paired with its unpermute; a fallback never consumes a TE map.

TE uses FP32 fixed-order summation where the original fallback uses BF16 atomic accumulation. Corrected Long counts may switch formerly rejected calls into TE; no bitwise claim against the old atomic full-training trajectory. The frozen combined implementation passed 372 operator comparisons and 9504 controlled FSDP comparisons. Historical full composition averaged 2.99248 / 2.99216 s over steps10–50; this PR alone is not a <3s claim.

Reorganized runtime validation on the qualified image: eight ranks each completed 50 normal-entry steps, exit 0, with the selected core paths active. Fixed steps10–50 averaged 3.014633 s/step across rank means; the full50-step average including startup was 3.635762 s/step, logged peak 71.85GB. This run did not reproduce the archived <3s result. No isolated contribution or long-run/resume claim is made.
