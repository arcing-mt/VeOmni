"""Extracted reference helpers from VeOmni commit 1adf524c; process-only."""
import os
import torch
from veomni.utils import logging
logger=logging.get_logger(__name__)
_ACE_TE_CACHE=[]
_ACE_TE_FELL_BACK=[]
last_fallback_reason=None
MERGE_CHECKS = os.environ.get('TE_MERGE_CHECKS', '0') == '1'
def _ace_te_impl():
    """Return ``(make_row_id_map, moe_permute_mask, moe_unpermute_mask, TE_DType)``, or ``None``.

    The ACE compaction runs either entirely on TransformerEngine kernels or
    entirely on the PyTorch fallback.  The two are driven by different maps -- TE
    by ``[num_local_experts, recv_tokens]`` expert-major ids, the fallback by the
    ``[topk, recv_tokens]`` slot order -- so a half-swapped pair would read the
    wrong rows.  ``VEOMNI_ACE_TE=0`` forces the fallback.
    """
    if _ACE_TE_CACHE:
        return _ACE_TE_CACHE[0]
    impl = None
    if os.environ.get('VEOMNI_ACE_TE', '1').strip().lower() not in {'0', 'false', 'no', 'off'}:
        try:
            from transformer_engine.pytorch import cpp_extensions as tex
            from transformer_engine.pytorch.constants import TE_DType
            from transformer_engine.pytorch.triton.permutation import make_row_id_map
            if hasattr(tex, 'moe_permute_mask') and hasattr(tex, 'moe_unpermute_mask'):
                impl = (make_row_id_map, tex.moe_permute_mask, tex.moe_unpermute_mask, TE_DType)
        except (ImportError, OSError):
            impl = None
    if impl is not None:
        logger.info_rank0('ACE compaction: using TransformerEngine kernels')
    _ACE_TE_CACHE.append(impl)
    return impl

def _ace_te_supported(recv_hidden, recv_probs, num_local_experts):
    """Whether the whole TE path can run on this payload."""
    if _ace_te_impl() is None:
        return False
    if not (recv_hidden.is_cuda or getattr(recv_hidden, 'is_musa', False)):
        return False
    if recv_hidden.dtype not in (torch.bfloat16, torch.float16):
        return False
    if not recv_hidden.is_contiguous():
        return False
    if recv_hidden.shape[-1] % (16 // recv_hidden.element_size()):
        return False
    return recv_probs.dtype == torch.float32 and num_local_experts % 4 == 0

class _TECompactPermute(torch.autograd.Function):
    """Gather the received rows into the expert-major layout with TE's permute.

    One fused kernel replaces the two ``index_select`` calls (activations and
    routing weights) of the PyTorch path.  Its adjoint is TE's un-permute: the
    autograd-generated adjoint of ``index_select`` accumulates duplicate rows with
    BF16 atomics, which on this workload measured 4.0e-3 against an FP64 reference
    and was not run-to-run reproducible, whereas TE sums in FP32 in a fixed order
    (6.9e-7, bit-reproducible).
    """

    @staticmethod
    def forward(ctx, recv_hidden, recv_probs, row_nt, row_map, num_tokens, width, num_out):
        _, tex_permute, _, te_dtype = _ace_te_impl()
        ctx.save_for_backward(row_nt, row_map)
        ctx.num_tokens = num_tokens
        ctx.width = width
        empty = torch.empty(0, device='cpu')
        return tex_permute(te_dtype[recv_hidden.dtype], recv_hidden, row_nt, recv_probs, num_tokens, width, num_out, recv_hidden.shape[-1], empty, empty)

    @staticmethod
    def backward(ctx, grad_output, grad_probs):
        _, _, tex_unpermute, te_dtype = _ace_te_impl()
        row_nt, row_map = ctx.saved_tensors
        empty = torch.empty(0, device='cpu')
        grad_hidden = tex_unpermute(te_dtype[grad_output.dtype], grad_output.contiguous(), row_map, empty, empty, ctx.num_tokens, ctx.width, grad_output.shape[-1], empty, empty)[0]
        if grad_probs is not None:
            grad_probs = grad_probs[row_nt.clamp_min(0)] * (row_nt >= 0).to(grad_probs.dtype)
        return (grad_hidden, grad_probs, None, None, None, None, None)

class _TECompactUnpermute(torch.autograd.Function):
    """Sum the expert-major rows back onto their received rows with TE kernels.

    The forward is TE's un-permute: one block per output row, the accumulator held
    in FP32 registers, contributions visited in a fixed order with ``-1`` slots
    skipped, so it never reads a padded row and is bit-reproducible.  The backward
    is TE's permute, the exact adjoint: measured against the ``index_select`` it
    replaces it is 31% faster (16-byte vector moves instead of int64-indexed row
    copies) and bit-identical to it.
    """

    @staticmethod
    def forward(ctx, weighted, row_id_map, width, num_tokens):
        _, _, tex_unpermute, te_dtype = _ace_te_impl()
        ctx.save_for_backward(row_id_map.t().contiguous())
        ctx.width = width
        ctx.num_tokens = num_tokens
        ctx.num_out = weighted.shape[0]
        empty = torch.empty(0, device='cpu')
        return tex_unpermute(te_dtype[weighted.dtype], weighted, row_id_map, empty, empty, num_tokens, width, weighted.shape[-1], empty, empty)[0]

    @staticmethod
    def backward(ctx, grad_output):
        _, tex_permute, _, te_dtype = _ace_te_impl()
        row_nt, = ctx.saved_tensors
        empty = torch.empty(0, device='cpu')
        grad_weighted = tex_permute(te_dtype[grad_output.dtype], grad_output.contiguous(), row_nt, torch.zeros_like(row_nt, dtype=torch.float32), ctx.num_tokens, ctx.width, ctx.num_out, grad_output.shape[-1], empty, empty)[0]
        return (grad_weighted, None, None, None)

def _te_compact_permute(recv_hidden, recv_indices, recv_probs, num_local_experts, expert_counts):
    """TE compaction: TE row ids plus one fused gather.

    Returns ``(permuted, probs, None, counts, row_id_map, num_local_experts)``, or
    ``None`` when this routing cannot take the TE path.
    """
    global last_fallback_reason
    last_fallback_reason = None
    make_row_id_map = _ace_te_impl()[0]
    num_tokens = recv_hidden.shape[0]
    multi_hot = torch.zeros((num_tokens, num_local_experts + 1), dtype=torch.bool, device=recv_hidden.device)
    multi_hot.scatter_(1, (recv_indices + 1).to(torch.int64), True)
    multi_hot = multi_hot[:, 1:].contiguous()
    counts = multi_hot.sum(0, dtype=torch.long)
    if MERGE_CHECKS:
        matches = (counts == torch.as_tensor(expert_counts, device=recv_hidden.device, dtype=torch.long)).all() if expert_counts is not None else counts.new_ones(())
        num_out, slot_total, counts_match = torch.stack([counts.sum(), (recv_indices >= 0).sum(), matches.to(torch.long)]).tolist()
    else:
        num_out, slot_total = torch.stack([counts.sum(), (recv_indices >= 0).sum()]).tolist()
        counts_match = None
    if num_out != slot_total:
        last_fallback_reason = 'duplicate_expert_slots'
        return None
    if expert_counts is not None and (not counts_match if MERGE_CHECKS else not torch.equal(counts, torch.as_tensor(expert_counts, device=recv_hidden.device, dtype=torch.long))):
        last_fallback_reason = 'expert_counts_mismatch'
        return None
    row_id_map, row_nt = make_row_id_map(multi_hot, num_tokens, num_local_experts)
    weights = torch.zeros((num_tokens, num_local_experts + 1), dtype=torch.float32, device=recv_hidden.device)
    weights.scatter_(1, (recv_indices + 1).to(torch.int64), recv_probs)
    weights = weights[:, 1:].contiguous()
    permuted, permuted_probs = _TECompactPermute.apply(recv_hidden, weights, row_nt, row_id_map, num_tokens, num_local_experts, num_out)
    return (permuted, permuted_probs, None, counts, row_id_map, num_local_experts)
