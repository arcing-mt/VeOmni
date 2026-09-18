"""Tests for the Qwen3.5-MoE MUSA dummy-forward skip patch.

The invariant under test is a hang-safety one: every rank must invoke the vision tower the
same number of times, or FSDP's gradient reduce-scatters desynchronise. The patch serves a
per-rank "budget" of `dummy_forward` calls instead of always serving both modality slots, so
the budget has to agree with what the other ranks expect — in every image/video mix, and
without short-circuiting the collective that decides it.

CPU-only: the distributed call is mocked.
"""

from unittest import mock

import pytest

from veomni.models.transformers.qwen3_5_moe import qwen3_5_moe_musa_runtime_patch as patch


def _budget(has_image, has_video, *, any_image, any_video):
    """Evaluate the planner from one rank's view of the cluster-wide answers."""
    calls = []

    def fake_all_reduce(data, op=None, group=None):
        calls.append((list(data), op, group))
        return [int(any_image), int(any_video)]

    with mock.patch.object(patch, "all_reduce", side_effect=fake_all_reduce):
        budget = patch._required_dummy_passes(has_image, has_video, group="mesh")
    return budget, calls


def _tower_passes(has_image, has_video, budget):
    """Tower invocations this rank ends up making: real slots plus the dummies it serves."""
    call_sites = int(not has_image) + int(not has_video)
    return int(has_image) + int(has_video) + min(budget, call_sites)


class TestCollectiveShape:
    def test_both_flags_travel_in_one_collective(self):
        """One all-reduce per rank, whatever the batch holds — no short-circuiting.

        A rank that skipped the collective when it already held a modality would issue a
        different number of collectives than a text-only rank, which deadlocks rather than
        merely disagreeing.
        """
        for has_image, has_video in [(True, True), (True, False), (False, True), (False, False)]:
            _, calls = _budget(has_image, has_video, any_image=True, any_video=True)
            assert len(calls) == 1, f"expected one collective for ({has_image}, {has_video}), got {calls}"
            local, op, group = calls[0]
            assert local == [int(has_image), int(has_video)]
            assert op == "max"
            assert group == "mesh"


class TestBudget:
    def test_image_only_dataset(self):
        """The optimisation: an image-only dataset costs one tower pass per rank.

        Rank 0 holds the image and serves no dummy; a text-only rank serves its image dummy
        only. Both end at one pass, and the video slot is skipped everywhere.
        """
        image_rank, _ = _budget(True, False, any_image=True, any_video=False)
        text_rank, _ = _budget(False, False, any_image=True, any_video=False)
        assert image_rank == 0
        assert text_rank == 1
        assert _tower_passes(True, False, image_rank) == 1
        assert _tower_passes(False, False, text_rank) == 1

    def test_all_text_batch_still_touches_the_tower(self):
        """No modality anywhere must still produce one pass per rank.

        The generated forward's unconditional dummy used to guarantee the tower was never
        left unused; multi-rank DDP (`fsdp_enabled` is a size check) fails on an unused
        trainable parameter, so the planner floors the count at one.
        """
        budget, _ = _budget(False, False, any_image=False, any_video=False)
        assert budget == 1
        assert _tower_passes(False, False, budget) == 1

    @pytest.mark.parametrize(
        ("has_image", "has_video"),
        [(True, True), (True, False), (False, True), (False, False)],
    )
    def test_both_modalities_present_keeps_two_passes(self, has_image, has_video):
        """A batch carrying both modalities behaves exactly like the unpatched forward."""
        budget, _ = _budget(has_image, has_video, any_image=True, any_video=True)
        assert _tower_passes(has_image, has_video, budget) == 2

    @pytest.mark.parametrize(
        ("has_image", "has_video", "any_image", "any_video"),
        [
            (True, False, True, False),
            (False, False, True, False),
            (False, True, False, True),
            (False, False, False, True),
        ],
    )
    def test_single_modality_cluster_keeps_one_pass(self, has_image, has_video, any_image, any_video):
        """Only one modality exists cluster-wide, so every rank runs the tower exactly once.

        The slot that exists is real on the rank holding it and a dummy elsewhere; the slot
        that exists nowhere is skipped on every rank. That is the optimisation.
        """
        budget, _ = _budget(has_image, has_video, any_image=any_image, any_video=any_video)
        assert _tower_passes(has_image, has_video, budget) == 1

    def test_all_text_batch_floors_at_one_pass(self):
        """The `any_* == 0` floor: one pass per rank rather than none."""
        budget, _ = _budget(False, False, any_image=False, any_video=False)
        assert budget == 1
        assert _tower_passes(False, False, budget) == 1

    def test_budget_never_exceeds_the_call_sites(self):
        """A leftover budget would leak into the next forward; it must always be consumable."""
        for has_image in (True, False):
            for has_video in (True, False):
                for any_image in (True, False):
                    for any_video in (True, False):
                        if has_image and not any_image or has_video and not any_video:
                            continue  # not reachable: a rank cannot hold what nobody holds
                        budget, _ = _budget(has_image, has_video, any_image=any_image, any_video=any_video)
                        assert budget <= int(not has_image) + int(not has_video)

    def test_ranks_agree_on_the_pass_count(self):
        """Exhaustive check of the invariant: equal tower passes across the group.

        Enumerates every assignment of modalities to a 3-rank group, keeps the assignments
        where all ranks report the same cluster-wide answers, and asserts the per-rank pass
        counts match — which is what the reduce-scatters need.
        """
        import itertools

        for assignment in itertools.product([(True, True), (True, False), (False, True), (False, False)], repeat=3):
            any_image = any(a for a, _ in assignment)
            any_video = any(v for _, v in assignment)
            counts = []
            for has_image, has_video in assignment:
                budget, _ = _budget(has_image, has_video, any_image=any_image, any_video=any_video)
                counts.append(_tower_passes(has_image, has_video, budget))
            assert len(set(counts)) == 1, f"{assignment} -> per-rank passes {counts}"
