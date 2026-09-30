"""Stable MUSA counting sort for DeepEP routing metadata."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _require_matching_counts(actual_counts: torch.Tensor, expected_counts: torch.Tensor) -> None:
    """Validate DeepEP counts without synchronizing an accelerator to Python."""
    if actual_counts.device.type == "cpu":
        if not torch.equal(actual_counts, expected_counts):
            raise RuntimeError("DeepEP expert counts do not match received routing slots")
        return

    torch._assert_async(
        (actual_counts == expected_counts).all(),
        "DeepEP expert counts do not match received routing slots",
    )


@triton.jit
def _count_experts_by_block(
    flat_experts,
    block_counts,
    num_slots,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block = tl.program_id(0)
    slot = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    expert = tl.load(flat_experts + slot, mask=slot < num_slots, other=-1)
    for expert_id in range(NUM_EXPERTS):
        count = tl.sum((expert == expert_id).to(tl.int32), axis=0)
        tl.store(block_counts + block * NUM_EXPERTS + expert_id, count)


@triton.jit
def _exclusive_prefix_by_expert(
    block_counts,
    block_prefix,
    num_blocks,
    NUM_EXPERTS: tl.constexpr,
):
    expert_id = tl.program_id(0)
    running = 0
    block = 0
    while block < num_blocks:
        offset = block * NUM_EXPERTS + expert_id
        count = tl.load(block_counts + offset)
        tl.store(block_prefix + offset, running)
        running += count
        block += 1


@triton.jit
def _scatter_stable_slots(
    flat_experts,
    expert_offsets,
    block_prefix,
    sorted_slots,
    token_rows,
    num_slots,
    TOP_K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block = tl.program_id(0)
    slot = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid_slot = slot < num_slots
    expert = tl.load(flat_experts + slot, mask=valid_slot, other=-1)
    for expert_id in range(NUM_EXPERTS):
        belongs = valid_slot & (expert == expert_id)
        local_rank = tl.cumsum(belongs.to(tl.int32), axis=0) - 1
        expert_offset = tl.load(expert_offsets + expert_id)
        block_offset = tl.load(block_prefix + block * NUM_EXPERTS + expert_id)
        destination = expert_offset + block_offset + local_rank
        tl.store(sorted_slots + destination, slot, mask=belongs)
        tl.store(token_rows + destination, slot // TOP_K, mask=belongs)


def musa_deepep_stable_slots(
    flat_experts: torch.Tensor,
    expert_counts: torch.Tensor,
    top_k: int,
    num_assignments: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return stable expert-major flat slots and their source-token rows.

    This is equivalent to ``nonzero`` followed by a stable ``argsort`` on the
    valid expert ids.  It uses the expert counts already returned by DeepEP to
    size the output and to establish each expert's output range.
    """
    if flat_experts.device.type != "musa":
        raise ValueError("the MUSA DeepEP compact kernel requires a MUSA tensor")
    if flat_experts.ndim != 1 or expert_counts.ndim != 1:
        raise ValueError("flat_experts and expert_counts must be 1-D")

    num_slots = flat_experts.numel()
    num_experts = expert_counts.numel()
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if num_assignments < 0 or num_assignments > num_slots:
        raise ValueError(f"num_assignments must be in [0, {num_slots}], got {num_assignments}")
    if num_slots == 0:
        _require_matching_counts(torch.zeros_like(expert_counts), expert_counts)
        empty = torch.empty(0, dtype=torch.int64, device=flat_experts.device)
        return empty, empty.clone()

    block_size = 1024
    num_blocks = triton.cdiv(num_slots, block_size)

    block_counts = torch.empty((num_blocks, num_experts), dtype=torch.int32, device=flat_experts.device)
    block_prefix = torch.empty_like(block_counts)
    # Allocate for every routing slot so inconsistent host counts can never
    # make the scatter write out of bounds before the asynchronous assertion
    # surfaces. The returned views retain only the expected valid assignments.
    sorted_slots = torch.empty(num_slots, dtype=torch.int64, device=flat_experts.device)
    token_rows = torch.empty_like(sorted_slots)

    _count_experts_by_block[(num_blocks,)](
        flat_experts,
        block_counts,
        num_slots,
        NUM_EXPERTS=num_experts,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    actual_counts = block_counts.sum(dim=0, dtype=torch.long)
    _require_matching_counts(actual_counts, expert_counts)
    expert_offsets = actual_counts.cumsum(0) - actual_counts
    _exclusive_prefix_by_expert[(num_experts,)](
        block_counts,
        block_prefix,
        num_blocks,
        NUM_EXPERTS=num_experts,
        num_warps=1,
    )
    _scatter_stable_slots[(num_blocks,)](
        flat_experts,
        expert_offsets,
        block_prefix,
        sorted_slots,
        token_rows,
        num_slots,
        TOP_K=top_k,
        NUM_EXPERTS=num_experts,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )
    return sorted_slots[:num_assignments], token_rows[:num_assignments]
