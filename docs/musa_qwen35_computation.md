---
orphan: true
---

# Opt-in MUSA Qwen3.5 training computation

Normal `tasks/train_vlm.py` / model loading now reaches these installed-package paths; no experimental launcher is required. Flags are process-wide and default to `0`.

| Flag (`=1`) | Training work removed | Fallback / limits |
| --- | --- | --- |
| `VEOMNI_MUSA_SUPERVISED_CE` | Project only supervised causal tokens into the vocabulary | Uses `chunk_loss`; SP, all/none supervised retain original path |
| `VEOMNI_MUSA_VISION_EMBEDDING_CSR` | Cache stable CSR indices for the vision positional embedding gradient | BF16 2304×1152, fixed grid helper only; hooks/DTensor/child FSDP use native embedding |
| `VEOMNI_MUSA_QK_PREENTRY` | Fold two-head repeat and native L2 normalization into one input read | Qualified packed Qwen3.5-MoE 16/32 heads, D128, no KV cache/Ulysses/compile; native cold autotune retained |

CSR/QK adapters verify the original source before any patch is installed. Generated modeling files remain unchanged. Unsupported source revisions fail clearly at opt-in build time. Per-call eligibility retains original behavior for unsupported shapes. No higher-order QK gradient claim. Optional Triton/FLA modules load only on selection.

Historical full composition (all three PR groups, TE MR156, same 8×S5000 50-step run): steps10–50 averaged 2.99248 and 2.99216 s/step. This is combined performance, not the independent gain of this PR. CE's earlier directional comparison was 3.4198 → 3.3411 s/step. Warmup/first-step costs are excluded from that steady window but retained in the full-run logs. Long-run convergence and resume equivalence are not certified.

Tests: full-projection CE loss/input+weight gradients, causal shift, mask boundaries, scaled upstream gradient, CPU/SP fallback, MUSA native QK parity.

Reorganized runtime validation on the qualified image: eight ranks each completed 50 normal-entry steps, exit 0, with the selected core paths active. Fixed steps10–50 averaged 3.014633 s/step across rank means; the full50-step average including startup was 3.635762 s/step, logged peak 71.85GB. This run did not reproduce the archived <3s result. No isolated contribution or long-run/resume claim is made.
