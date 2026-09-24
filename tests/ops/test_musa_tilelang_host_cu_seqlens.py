# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host-side packed offsets for the ``musa_tilelang`` GDN adapter.

The adapter normally learns the per-sequence lengths from the device
``cu_seqlens``, which costs one blocking device-to-host sync per layer per forward
(``torch_kernels``' ``varlen_lengths``). Callers that already hold those offsets on
the host pass them in instead. Three invariants keep that safe, and each has a
test here:

* the capability probe reports what the *installed* ``torch_kernels`` accepts, so
  a build without ``cu_seqlens_cpu`` degrades to the old sync instead of raising;
* the offsets are only forwarded when both the probe and the caller say so;
* offsets that do not describe the same segmentation as ``cu_seqlens`` fail loudly
  -- the vendor only checks that they sum to the packed length, so a same-total
  pair with different boundaries would otherwise resegment silently.

CPU-only: the vendor entry point is stubbed, no device or ``torch_kernels`` install
is required.
"""

import functools
from contextlib import nullcontext
from unittest import mock

import pytest
import torch

from veomni.ops.kernels.gated_delta_rule import musa_tilelang


def _accepts_cu_seqlens_cpu(
    q,
    k,
    v,
    g,
    beta,
    *,
    scale=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
):
    """Stand-in for a patched ``torch_kernels`` GDN entry point."""


def _accepts_cu_seqlens_cpu_kwonly(q, k, v, g, beta, *, cu_seqlens=None, cu_seqlens_cpu=None):
    """Same, with the offset keyword after the ``*``."""


def _old_abi(
    q,
    k,
    v,
    g,
    beta,
    *,
    scale=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    cu_seqlens=None,
):
    """Stand-in for a ``torch_kernels`` build that predates the parameter."""


def _local_only():
    """A function that merely *mentions* the name in its body must not count."""


class TestCapabilityProbe:
    def test_detects_the_parameter(self):
        assert musa_tilelang._probe_accepts_host_cu_seqlens(_accepts_cu_seqlens_cpu) is True

    def test_detects_a_keyword_only_parameter(self):
        assert musa_tilelang._probe_accepts_host_cu_seqlens(_accepts_cu_seqlens_cpu_kwonly) is True

    def test_rejects_a_build_without_the_parameter(self):
        assert musa_tilelang._probe_accepts_host_cu_seqlens(_old_abi) is False

    def test_rejects_a_local_name_that_is_not_a_parameter(self):
        # ``co_varnames`` also holds locals, which is why the probe slices it to the
        # argument count: a stray local must not be mistaken for the parameter.
        def mentions_it(q, k, v, g, beta, *, cu_seqlens=None):
            cu_seqlens_cpu = None  # noqa: F841 - the point of the test
            return cu_seqlens

        assert musa_tilelang._probe_accepts_host_cu_seqlens(mentions_it) is False

    def test_rejects_a_callable_it_cannot_introspect(self):
        assert musa_tilelang._probe_accepts_host_cu_seqlens(functools.partial(_old_abi)) is False


class TestForwarding:
    """The adapter forwards the offsets only when both gates agree."""

    def _records(self):
        calls = []

        def fake_gated_delta_net(q, k, v, g, beta, **kwargs):
            calls.append(kwargs)
            return v, None

        return calls, fake_gated_delta_net

    def _run(self, cu_seqlens_cpu, *, supported, contextvar=None):
        calls, fake = self._records()
        cu_seqlens = torch.tensor([0, 4, 7], dtype=torch.int32)
        with (
            mock.patch.object(musa_tilelang, "_require_torch_kernels", return_value=fake),
            mock.patch.object(musa_tilelang, "_accepts_host_cu_seqlens", return_value=supported),
        ):
            enter = musa_tilelang.forward_cu_seqlens_cpu(contextvar) if contextvar is not None else nullcontext()
            with enter:
                musa_tilelang.chunk_gated_delta_rule(
                    torch.zeros(1, 7, 2, 8),
                    torch.zeros(1, 7, 2, 8),
                    torch.zeros(1, 7, 2, 8),
                    torch.zeros(1, 7, 2),
                    torch.zeros(1, 7, 2),
                    cu_seqlens=cu_seqlens,
                    cu_seqlens_cpu=cu_seqlens_cpu,
                )
        return calls[0]

    def test_forwards_when_supported(self):
        assert self._run([0, 4, 7], supported=True)["cu_seqlens_cpu"] == [0, 4, 7]

    def test_drops_when_the_installed_torch_kernels_is_too_old(self):
        """The old build must not receive a keyword it would reject."""
        assert "cu_seqlens_cpu" not in self._run([0, 4, 7], supported=False)

    def test_drops_when_the_caller_has_no_offsets(self):
        assert "cu_seqlens_cpu" not in self._run(None, supported=True)

    def test_reads_the_scoped_offsets_when_the_argument_is_absent(self):
        assert self._run(None, supported=True, contextvar=[0, 4, 7])["cu_seqlens_cpu"] == (0, 4, 7)


class TestSegmentationGuard:
    """A mismatched pair of offsets is a loud failure, not a silent resegment."""

    def _call(self, host_offsets):
        def fake_gated_delta_net(q, k, v, g, beta, **kwargs):
            raise AssertionError("the vendor must not be reached with mismatched offsets")

        with (
            mock.patch.object(musa_tilelang, "_require_torch_kernels", return_value=fake_gated_delta_net),
            mock.patch.object(musa_tilelang, "_accepts_host_cu_seqlens", return_value=True),
        ):
            musa_tilelang.chunk_gated_delta_rule(
                torch.zeros(1, 7, 2, 8),
                torch.zeros(1, 7, 2, 8),
                torch.zeros(1, 7, 2, 8),
                torch.zeros(1, 7, 2),
                torch.zeros(1, 7, 2),
                cu_seqlens=torch.tensor([0, 4, 7], dtype=torch.int32),
                cu_seqlens_cpu=host_offsets,
            )

    def test_entry_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="same segmentation"):
            self._call([0, 7])

    def test_matching_entry_count_passes_the_guard(self):
        # Same total, different boundaries: the vendor catches this one (it validates
        # strict monotonicity and the packed length), so reaching it is correct.
        with pytest.raises(AssertionError, match="must not be reached"):
            self._call([0, 3, 7])

    def test_guard_does_not_read_the_device(self):
        """The check compares entry counts, so it must never touch tensor values."""
        cu_seqlens = mock.MagicMock()
        cu_seqlens.numel.return_value = 3

        def vendor(*args, **kwargs):
            raise AssertionError("the vendor must not be reached with mismatched offsets")

        with (
            mock.patch.object(musa_tilelang, "_require_torch_kernels", return_value=vendor),
            mock.patch.object(musa_tilelang, "_accepts_host_cu_seqlens", return_value=True),
        ):
            with pytest.raises(ValueError, match="same segmentation"):
                musa_tilelang.chunk_gated_delta_rule(
                    torch.zeros(1, 7, 2, 8),
                    torch.zeros(1, 7, 2, 8),
                    torch.zeros(1, 7, 2, 8),
                    torch.zeros(1, 7, 2),
                    torch.zeros(1, 7, 2),
                    cu_seqlens=cu_seqlens,
                    cu_seqlens_cpu=[0, 7],
                )
        assert not cu_seqlens.tolist.called
        assert not cu_seqlens.__getitem__.called


class TestContextVar:
    def test_none_outside_and_restored_after(self):
        assert musa_tilelang._forward_cu_seqlens_cpu.get() is None
        with musa_tilelang.forward_cu_seqlens_cpu([0, 4]):
            assert musa_tilelang._forward_cu_seqlens_cpu.get() == (0, 4)
            with musa_tilelang.forward_cu_seqlens_cpu([0, 9]):
                assert musa_tilelang._forward_cu_seqlens_cpu.get() == (0, 9)
            assert musa_tilelang._forward_cu_seqlens_cpu.get() == (0, 4)
        assert musa_tilelang._forward_cu_seqlens_cpu.get() is None

    def test_restored_when_the_body_raises(self):
        with pytest.raises(RuntimeError):
            with musa_tilelang.forward_cu_seqlens_cpu([0, 4]):
                raise RuntimeError("forward blew up")
        assert musa_tilelang._forward_cu_seqlens_cpu.get() is None
