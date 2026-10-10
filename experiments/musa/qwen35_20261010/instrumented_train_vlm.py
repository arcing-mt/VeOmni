"""Measure complete train steps without adding synchronization or changing math."""
import functools
import json
import os
from pathlib import Path
import runpy
import sys
import time

from veomni.trainer.vlm_trainer import VLMTrainer
from veomni.ops.dispatch import OpSlot

if os.environ.get("QK_PREENTRY", "0") == "1":
    import install_qk_preentry
    install_qk_preentry.install()

if os.environ.get("FLA_CONV_VJP", "0") == "1":
    import install_conv_vjp
    install_conv_vjp.install()

if os.environ.get("HOST_CU_CACHE", "0") == "1":
    import metadata_cache
    metadata_cache.install()

if os.environ.get("FUSED_UNPERMUTE", "0") == "1":
    import install_fused_unpermute
    install_fused_unpermute.install()

if os.environ.get("SCALAR_COMPACT", "0") == "1":
    import install_scalar_compact
    install_scalar_compact.install()

if os.environ.get("TE_COMPACT", "0") == "1":
    import install_te_compact
    install_te_compact.install()

if os.environ.get("TE_FUSED_UNPERMUTE", "0") == "1":
    if os.environ.get("TE_COMPACT", "0") != "1":
        raise RuntimeError("TE_FUSED_UNPERMUTE requires TE_COMPACT=1.")
    import install_te_fused_unpermute
    install_te_fused_unpermute.install()

if os.environ.get("SUPERVISED_CE", "0") == "1":
    import supervised_chunk_loss
    supervised_chunk_loss.install()

if os.environ.get("SUPERVISED_CE_CHUNK2048", "0") == "1":
    if os.environ.get("SUPERVISED_CE", "0") != "1":
        raise RuntimeError("CE chunk2048 trial requires the validated compact CE.")
    import install_supervised_chunk_size
    install_supervised_chunk_size.install()

if os.environ.get("MCCL_SHARD_PADDING", "0") == "1":
    import install_mccl_padding
    install_mccl_padding.install()

if os.environ.get("FSDP_FUSED_AG_PACK", "0") == "1":
    import fsdp_fused_ag_pack
    fsdp_fused_ag_pack.install()
    fsdp_fused_ag_pack.enabled = True

if os.environ.get("FSDP_ALIGNED_AG_PACK", "0") == "1":
    if os.environ.get("FSDP_FUSED_AG_PACK", "0") == "1":
        raise RuntimeError("Choose only one AG pack experiment")
    import fsdp_aligned_ag_pack
    fsdp_aligned_ag_pack.install()
    fsdp_aligned_ag_pack.enabled = True

if os.environ.get("DEFERRED_ADAM_CLIP", "0") == "1":
    import install_deferred_clip
    install_deferred_clip.install()

if os.environ.get("VISION_RETAIN_THIRD", "0") == "1":
    import install_vision_checkpoint_policy
    install_vision_checkpoint_policy.install()

if os.environ.get("TEXT_RETAIN_EVERY5", "0") == "1":
    if os.environ.get("VISION_RETAIN_THIRD", "0") == "1":
        raise RuntimeError("Text retention trial requires the original Vision policy.")
    import install_text_checkpoint_policy
    install_text_checkpoint_policy.install()

if os.environ.get("EARLY_RECURRENT_MASK", "0") == "1":
    import install_early_recurrent_mask
    install_early_recurrent_mask.install()

if os.environ.get("DEFER_TRAIN_SCALARS", "0") == "1":
    if os.environ.get("DEFERRED_ADAM_CLIP", "0") == "1":
        raise RuntimeError("Deferred reporting trial requires the original clip/Adam path.")
    import install_deferred_train_scalars
    install_deferred_train_scalars.install()

if os.environ.get("FSDP_PREFETCH_DEPTH", "1") == "2":
    import install_fsdp_prefetch_depth
    install_fsdp_prefetch_depth.install()

if os.environ.get("VISION_EMBEDDING_CSR", "0") == "1":
    import install_vision_embedding_csr
    install_vision_embedding_csr.install()

if os.environ.get("VISION_GEOMETRY_CACHE", "0") == "1":
    import install_vision_geometry_cache
    install_vision_geometry_cache.install()

original = VLMTrainer.train_step
samples = []
op_metadata = {}
original_bind = OpSlot.bind


def capture_bind(self, implementation):
    result = original_bind(self, implementation)
    if self.op_name not in {"chunk_gated_delta_rule", "causal_conv1d", "rms_norm_gated"} or self._kernel is None:
        return result
    kernel = self._kernel

    @functools.wraps(kernel)
    def capture(*args, **kwargs):
        metadata = {}
        for key, value in list(enumerate(args)) + list(kwargs.items()):
            if hasattr(value, "shape") and hasattr(value, "dtype"):
                metadata[str(key)] = {"shape": list(value.shape), "dtype": str(value.dtype), "stride": list(value.stride())}
        key = json.dumps([self.op_name, implementation, metadata], sort_keys=True)
        op_metadata[key] = op_metadata.get(key, 0) + 1
        return kernel(*args, **kwargs)

    self._kernel = capture
    return result


if os.environ.get("CAPTURE_OP_METADATA", "0") == "1":
    OpSlot.bind = capture_bind


@functools.wraps(original)
def measured(self, *args, **kwargs):
    # The original method includes data fetch, forward/backward, optimizer and
    # callback metric reductions. Those existing reductions synchronize work.
    diagnostic = None
    if os.environ.get("LIGHT_ROUTING_DIAGNOSTIC", "0") == "1":
        import routing_light as diagnostic
        diagnostic.begin(self)
    elif os.environ.get("PAIRED_ROUTING_DIAGNOSTIC", "0") == "1":
        import routing_diagnostic as diagnostic
        diagnostic.begin(self)
    start = time.perf_counter()
    try:
        result = original(self, *args, **kwargs)
    finally:
        if diagnostic is not None:
            diagnostic.flush()
    elapsed = time.perf_counter() - start
    samples.append({"step": self.base.state.global_step, "seconds": elapsed})
    return result


VLMTrainer.train_step = measured
try:
    runpy.run_path(str(Path.cwd() / "tasks/train_vlm.py"), run_name="__main__")
finally:
    training_had_error = sys.exc_info()[0] is not None
    import torch
    rank = os.environ.get("RANK", "unknown")
    target = Path(os.environ["STEP_TIMING_DIR"]) / f"step_times_rank{rank}.json"
    target.write_text(json.dumps({"rank": rank, "scope": "complete VLMTrainer.train_step; no added device synchronization", "samples": samples}, indent=2) + "\n")
    if hasattr(torch, "musa") and torch.musa.is_available():
        (target.parent / f"memory_rank{rank}.json").write_text(json.dumps({
            "peak_allocated_bytes": torch.musa.max_memory_allocated(),
            "peak_reserved_bytes": torch.musa.max_memory_reserved(),
            "allocated_bytes": torch.musa.memory_allocated(),
            "reserved_bytes": torch.musa.memory_reserved(),
            "scope": "PyTorch allocator counters after training; external allocations excluded"
        }, indent=2) + "\n")
    if os.environ.get("EARLY_RECURRENT_MASK", "0") == "1":
        (target.parent / f"early_mask_rank{rank}.json").write_text(json.dumps(install_early_recurrent_mask.stats(), indent=2) + "\n")
    if os.environ.get("DEFER_TRAIN_SCALARS", "0") == "1":
        (target.parent / f"deferred_scalars_rank{rank}.json").write_text(json.dumps(install_deferred_train_scalars.stats(), indent=2) + "\n")
    if os.environ.get("FSDP_PREFETCH_DEPTH", "1") == "2":
        (target.parent / f"prefetch_depth_rank{rank}.json").write_text(json.dumps(install_fsdp_prefetch_depth.stats(), indent=2) + "\n")
    if os.environ.get("VISION_EMBEDDING_CSR", "0") == "1":
        (target.parent / f"vision_embedding_rank{rank}.json").write_text(json.dumps(install_vision_embedding_csr.stats(), indent=2) + "\n")
    if os.environ.get("VISION_GEOMETRY_CACHE", "0") == "1":
        (target.parent / f"vision_geometry_rank{rank}.json").write_text(json.dumps(install_vision_geometry_cache.stats(), indent=2) + "\n")
    if os.environ.get("HOST_CU_CACHE", "0") == "1":
        (target.parent / f"metadata_cache_rank{rank}.json").write_text(json.dumps(metadata_cache.stats(), indent=2) + "\n")
    if os.environ.get("SUPERVISED_CE_CHUNK2048", "0") == "1":
        (target.parent / f"supervised_chunk_size_rank{rank}.json").write_text(json.dumps(install_supervised_chunk_size.stats(), indent=2) + "\n")
    if os.environ.get("FUSED_UNPERMUTE", "0") == "1":
        (target.parent / f"fused_unpermute_rank{rank}.json").write_text(json.dumps(install_fused_unpermute.stats(), indent=2) + "\n")
    if os.environ.get("SCALAR_COMPACT", "0") == "1":
        (target.parent / f"scalar_compact_rank{rank}.json").write_text(json.dumps(install_scalar_compact.stats(), indent=2) + "\n")
    if os.environ.get("TE_COMPACT", "0") == "1":
        (target.parent / f"te_compact_rank{rank}.json").write_text(json.dumps(install_te_compact.stats(), indent=2) + "\n")
    if os.environ.get("TE_FUSED_UNPERMUTE", "0") == "1":
        (target.parent / f"te_fused_unpermute_rank{rank}.json").write_text(json.dumps(install_te_fused_unpermute.stats(), indent=2) + "\n")
    if os.environ.get("SUPERVISED_CE", "0") == "1":
        (target.parent / f"supervised_ce_rank{rank}.json").write_text(json.dumps(supervised_chunk_loss.stats(), indent=2) + "\n")
    if os.environ.get("TEXT_RETAIN_EVERY5", "0") == "1":
        (target.parent / f"text_checkpoint_rank{rank}.json").write_text(json.dumps(install_text_checkpoint_policy.stats(), indent=2) + "\n")
    if os.environ.get("MCCL_SHARD_PADDING", "0") == "1":
        (target.parent / f"mccl_padding_rank{rank}.json").write_text(json.dumps(install_mccl_padding.stats(), indent=2) + "\n")
    if os.environ.get("FSDP_FUSED_AG_PACK", "0") == "1":
        (target.parent / f"fsdp_ag_pack_rank{rank}.json").write_text(json.dumps(fsdp_fused_ag_pack.stats(), indent=2) + "\n")
    if os.environ.get("FSDP_ALIGNED_AG_PACK", "0") == "1":
        (target.parent / f"fsdp_aligned_ag_pack_rank{rank}.json").write_text(json.dumps(fsdp_aligned_ag_pack.stats(), indent=2) + "\n")
    if os.environ.get("DEFERRED_ADAM_CLIP", "0") == "1":
        (target.parent / f"deferred_clip_rank{rank}.json").write_text(json.dumps(install_deferred_clip.stats(), indent=2) + "\n")
    if os.environ.get("VISION_RETAIN_THIRD", "0") == "1":
        (target.parent / f"vision_checkpoint_rank{rank}.json").write_text(json.dumps(install_vision_checkpoint_policy.stats(), indent=2) + "\n")
    if os.environ.get("CAPTURE_OP_METADATA", "0") == "1":
        selected = {}
        for module_name in ("fla.modules.l2norm", "fla.modules.fused_norm_gate", "fla.modules.conv.triton.kernels"):
            module = sys.modules.get(module_name)
            if module is None:
                continue
            for name, kernel in vars(module).items():
                current = kernel
                for _ in range(8):
                    if hasattr(current, "configs"):
                        if getattr(current, "cache", None):
                            selected[f"{module_name}.{name}"] = {str(k): str(v) for k,v in current.cache.items()}
                        break
                    if not hasattr(current, "fn"):
                        break
                    current = current.fn
        (target.parent / f"op_metadata_rank{rank}.json").write_text(json.dumps({"calls": op_metadata, "selected_fla_configs": selected}, indent=2) + "\n")
    if os.environ.get("QK_PREENTRY", "0") == "1":
        try:
            qk_stats = install_qk_preentry.stats()
            qk_stats["runtime_deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
            qk_stats["status"] = "passed" if qk_stats["prototype"].get("fused_backward", 0) > 0 else "failed_no_hit"
            (target.parent / f"qk_preentry_rank{rank}.json").write_text(json.dumps(qk_stats, indent=2) + "\n")
            if qk_stats["status"] != "passed":
                raise RuntimeError("QK_PREENTRY requested but no eligible backward ran")
        except Exception as error:
            if not training_had_error:
                raise
            print(f"Post-training q/k diagnostics failed; original training error preserved: {error}", file=sys.stderr)
    if os.environ.get("FLA_CONV_VJP", "0") == "1":
        conv_stats = install_conv_vjp.stats()
        conv_stats["status"] = "passed" if conv_stats["backward"].get("fused_bwd", 0) > 0 else "failed_no_hit"
        try:
            (target.parent / f"conv_vjp_rank{rank}.json").write_text(json.dumps(conv_stats, indent=2) + "\n")
            if conv_stats["status"] != "passed":
                raise RuntimeError("FLA_CONV_VJP requested but no eligible backward ran")
        except Exception as error:
            if not training_had_error:
                raise
            print(f"Post-training conv diagnostics failed; original training error preserved: {error}", file=sys.stderr)
    if os.environ.get("SHARED_SCHEDULE_DIAGNOSTIC", "0") == "1":
        import inspect
        diagnostic = {"expected_overlap": os.environ.get("MOE_SHARED_EXPERT_OVERLAP", "true") == "true",
                      "loaded_modules": [],
                      "scope": "Post-training CPU-only loaded-class check; no measured-step hooks or added GPU synchronization."}
        diagnostic_failure = None
        try:
            for name, module in list(sys.modules.items()):
                if not name.startswith("veomni.models.transformers.qwen3_5_moe.") or module is None:
                    continue
                cls = vars(module).get("Qwen3_5MoeSparseMoeBlock")
                if cls is None:
                    continue
                diagnostic["loaded_modules"].append({
                    "module": name,
                    "overlap_patch_installed": bool(vars(module).get("_VEOMNI_MUSA_SHARED_EXPERT_OVERLAP_PATCHED", False)),
                    "forward_qualname": cls.forward.__qualname__,
                    "forward_source": inspect.getsourcefile(cls.forward)})
            if not diagnostic["loaded_modules"]:
                raise RuntimeError("Shared schedule diagnostic did not find the loaded Qwen sparse block")
            if any(row["overlap_patch_installed"] != diagnostic["expected_overlap"] for row in diagnostic["loaded_modules"]):
                raise RuntimeError("Loaded shared scheduling path does not match flag")
            diagnostic["status"] = "passed"
        except Exception as error:
            diagnostic_failure = error
            diagnostic.update(status="failed", error=repr(error))
        try:
            (target.parent / f"shared_schedule_rank{rank}.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
        except Exception as error:
            diagnostic_failure = error
        if diagnostic_failure is not None:
            if not training_had_error:
                raise RuntimeError("Post-training shared schedule diagnostic failed") from diagnostic_failure
            print(f"Shared schedule diagnostic also failed: {diagnostic_failure!r}", file=sys.stderr)
