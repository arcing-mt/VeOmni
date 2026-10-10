"""Process-only TE compaction; restore the host Torch APIs after TE import."""
import json
import sys
from collections import Counter
import torch
import triton
from veomni.distributed.moe import deepep_ace
import te_compact_prototype as reference

calls = 0
fallbacks = 0
shapes = Counter()
fallback_reasons = Counter()


def load_native():
    if reference._ACE_TE_CACHE:
        return reference._ACE_TE_CACHE[0]
    if 'transformer_engine' in sys.modules:
        raise RuntimeError('TE was already imported; its original Torch API state cannot be recovered by this trial.')
    surfaces = (torch,torch.Tensor,torch.nn.Module,torch.cuda,torch.cuda.nvtx,torch.distributed)
    originals = [(surface,dict(vars(surface))) for surface in surfaces]
    try:
        from transformer_engine.pytorch import cpp_extensions as tex
        from transformer_engine.pytorch.constants import TE_DType
        from transformer_engine.pytorch.triton.permutation import _row_id_map_pass_1_kernel, _row_id_map_pass_2_kernel
    finally:
        # TE's MUSA compatibility layer rewrites global Torch functions. The
        # native kernels used here need none of those Python-level aliases.
        for surface,attributes in originals:
            current = vars(surface)
            for name,value in attributes.items():
                if current.get(name) is not value:
                    setattr(surface,name,value)

    def make_map(routing,tokens,experts):
        rows = torch.empty((experts,tokens),device=routing.device,dtype=torch.int64)
        transposed = torch.empty((tokens,experts),device=routing.device,dtype=torch.int64)
        block = 256
        grid = (experts,triton.cdiv(tokens,block))
        workspace = torch.empty(grid,device=routing.device,dtype=torch.int64)
        _row_id_map_pass_1_kernel[grid](routing,rows,workspace,tokens,routing.stride(0),routing.stride(1),block)
        _row_id_map_pass_2_kernel[grid](rows,transposed,workspace,experts,tokens,triton.next_power_of_2(experts*grid[1]),block)
        return rows,transposed

    reference._ACE_TE_CACHE[:] = [(make_map,tex.moe_permute_mask,tex.moe_unpermute_mask,TE_DType)]
    return reference._ACE_TE_CACHE[0]


def install():
    load_native()
    original_permute = deepep_ace._compact_permute
    original_unpermute = deepep_ace._compact_unpermute

    def permute(hidden,indices,probabilities,experts,expert_counts=None):
        global fallbacks,calls
        if (hidden.shape[0] == 0 or (expert_counts is not None and sum(expert_counts) == 0)
                or (hidden.device.type == 'musa' and hidden.device.index != torch.musa.current_device())
                or not reference._ace_te_supported(hidden,probabilities,experts)):
            fallbacks += 1
            reason = ('zero_recv_tokens' if hidden.shape[0] == 0 else
                      'zero_assignments' if expert_counts is not None and sum(expert_counts) == 0 else
                      'device_or_layout_unsupported')
            fallback_reasons[reason] += 1
            return original_permute(hidden,indices,probabilities,experts,expert_counts)
        value = reference._te_compact_permute(hidden,indices,probabilities,experts,expert_counts)
        if value is None:
            fallbacks += 1
            fallback_reasons[reference.last_fallback_reason or 'unknown_reference_rejection'] += 1
            return original_permute(hidden,indices,probabilities,experts,expert_counts)
        y,p,_,counts,row_map,width = value
        if y.shape[0] == 0:
            # The native API's zero-token backward is not part of this trial.
            fallbacks += 1
            fallback_reasons['zero_native_output'] += 1
            return original_permute(hidden,indices,probabilities,experts,expert_counts)
        token_rows = torch.empty(0,device=hidden.device,dtype=torch.long)
        token_rows._te_compaction = (row_map,width)
        calls += 1
        shapes[json.dumps([hidden.shape[0],hidden.shape[1],y.shape[0],experts])] += 1
        return y,p,token_rows,counts

    def unpermute(y,p,rows,recv_tokens):
        metadata = getattr(rows,'_te_compaction',None)
        if metadata is None:
            return original_unpermute(y,p,rows,recv_tokens)
        row_map,width = metadata
        weighted = y * p.to(y.dtype).unsqueeze(-1)
        return reference._TECompactUnpermute.apply(weighted,row_map,width,recv_tokens)

    deepep_ace._compact_permute = permute
    deepep_ace._compact_unpermute = unpermute


def stats():
    return {'calls':calls,'fallbacks':fallbacks,'fallback_reasons':dict(fallback_reasons),'shapes':dict(shapes),
            'merged_checks': reference.MERGE_CHECKS,
            'note':'TE native compaction; all map/check costs included; FP32 fixed accumulation; global Torch APIs restored.'}
