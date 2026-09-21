"""Tests for the MUSA batched FSDP2 gradient-norm plugin.

The plugin rebinds two private helpers of the shared clip module, so these tests
check behaviour rather than the shape of the implementation: agreement with an
independent reference, agreement with the shared implementation it replaces, and
that the rebinding is reversible.
"""

import contextlib
import importlib
import tempfile

import pytest
import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, Shard
from torch.distributed.device_mesh import DeviceMesh

from veomni.utils.device import IS_MUSA_AVAILABLE, get_device_type


# Both packages re-export a *function* named after the submodule, so reach the
# modules through importlib instead of ``from ... import``.
clip_grad_norm_module = importlib.import_module("veomni.distributed.fsdp2.clip_grad_norm")
plugin = importlib.import_module("veomni.ops.platform.musa.fsdp2_clip_grad_norm")


pytestmark = pytest.mark.skipif(
    not IS_MUSA_AVAILABLE,
    reason="the MUSA clip plugin is only installed on MUSA, and torch_musa's empty-tensor max() differs from CPU/CUDA",
)

DEVICE = get_device_type()

# The shared spellings, captured before any test installs the plugin.
SHARED_PTH_SUM = clip_grad_norm_module._local_pth_sum
SHARED_MAX = clip_grad_norm_module._local_max


@contextlib.contextmanager
def patched():
    """Install the plugin and restore both modules on exit."""
    originals = (clip_grad_norm_module._local_pth_sum, clip_grad_norm_module._local_max)
    was_patched = plugin._PATCHED
    plugin._PATCHED = False
    try:
        plugin.apply_musa_fsdp2_clip_grad_norm_patch()
        yield
    finally:
        plugin.revert_musa_fsdp2_clip_grad_norm_patch()
        clip_grad_norm_module._local_pth_sum, clip_grad_norm_module._local_max = originals
        plugin._PATCHED = was_patched


def _params_from_specs(specs, dtype=torch.float32, device=None):
    """Build parameters from ``(numel, fill_value)`` pairs; ``None`` leaves the grad unset."""
    device = device or DEVICE
    params = []
    for size, fill in specs:
        param = torch.nn.Parameter(torch.zeros(size, dtype=dtype, device=device))
        param.grad = None if fill is None else torch.full((size,), float(fill), dtype=dtype, device=device)
        params.append(param)
    return params


def _random_params(dtypes=(torch.bfloat16,), devices=None, per_pair=6, drop_grad_every=3):
    """Build ``per_pair`` parameters for every (device, dtype) pair."""
    params = []
    index = 0
    for device in devices or (DEVICE,):
        for dtype in dtypes:
            for _ in range(per_pair):
                size = 1 + index * 37 % 129
                param = torch.nn.Parameter(torch.zeros(size, dtype=dtype, device=device))
                if drop_grad_every and index % drop_grad_every == 0:
                    param.grad = None
                else:
                    grad = torch.randn(size, dtype=torch.float32) * 0.05
                    param.grad = grad.to(dtype=dtype, device=device)
                params.append(param)
                index += 1
    return params


def _reference_pth_sum(params, p):
    """Sum of per-tensor p-th powers, computed the slow, obviously-correct way."""
    total = torch.tensor(0.0, dtype=torch.float64)
    for param in params:
        if param.grad is None:
            continue
        grad = param.grad.detach().to(device="cpu", dtype=torch.float64)
        total = total + torch.linalg.vector_norm(grad, ord=p).pow(p)
    return total


def _reference_max(params):
    values = [
        torch.abs(param.grad.detach().to(device="cpu", dtype=torch.float64)).max()
        for param in params
        if param.grad is not None
    ]
    if not values:
        return torch.tensor(0.0, dtype=torch.float64)
    return torch.stack(values).max()


@pytest.mark.parametrize("p", [0.0, 1.0, 2.0, 3.5])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_batched_pth_sum_matches_reference(dtype, p):
    params = _random_params(dtypes=(dtype,))

    actual = plugin.musa_local_pth_sum(params, p).cpu().double()

    torch.testing.assert_close(actual, _reference_pth_sum(params, p), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_batched_max_matches_reference(dtype):
    params = _random_params(dtypes=(dtype,))

    actual = plugin.musa_local_max(params).cpu().double()

    torch.testing.assert_close(actual, _reference_max(params), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("p", [0.0, 1.0, 2.0, 3.5, float("inf")])
def test_batched_helpers_agree_with_the_shared_implementation(p):
    """The plugin must not change the value the shared implementation produces."""
    params = _random_params(dtypes=(torch.float16, torch.bfloat16, torch.float32, torch.float64))

    expected = SHARED_PTH_SUM(params, p)
    with patched():
        actual = clip_grad_norm_module._local_pth_sum(params, p)

    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-5, atol=1e-6)


def test_batched_max_agrees_with_the_shared_implementation():
    params = _random_params(dtypes=(torch.float16, torch.bfloat16, torch.float32))

    expected = SHARED_MAX(params)
    with patched():
        actual = clip_grad_norm_module._local_max(params)

    assert torch.equal(actual.cpu(), expected.cpu())


@pytest.mark.parametrize("norm_type", [1.0, 2.0, float("inf")])
def test_every_foreach_call_receives_one_device_and_dtype(monkeypatch, norm_type):
    """``_foreach_norm`` silently degrades to one reduction per tensor unless each
    call's list is homogeneous in device *and* dtype, so the bucketing is the
    property that decides whether this plugin is fast or merely correct."""
    params = [
        *_params_from_specs([(4, 1.0)], dtype=torch.bfloat16),
        *_params_from_specs([(4, 1.0)], dtype=torch.float16),
        *_params_from_specs([(4, 1.0)], dtype=torch.float32),
    ]
    calls = []
    real_foreach_norm = torch._foreach_norm

    def spy(tensors, ord, dtype=None):
        calls.append(list(tensors))
        return real_foreach_norm(tensors, ord, dtype=dtype)

    monkeypatch.setattr(torch, "_foreach_norm", spy)

    if norm_type == float("inf"):
        plugin.musa_local_max(params)
    else:
        plugin.musa_local_pth_sum(params, norm_type)

    assert len(calls) == 3, "one call per (device, dtype) bucket"
    for tensors in calls:
        assert len({(t.device.type, t.dtype) for t in tensors}) == 1


def test_unbatchable_order_still_returns_the_right_value():
    """``norm_type=3.5`` has no batched kernel; the value must survive the fallback."""
    params = _random_params(dtypes=(torch.bfloat16,), per_pair=4, drop_grad_every=0)
    p = 3.5

    actual = plugin.musa_local_pth_sum(params, p).cpu().double()

    torch.testing.assert_close(actual, _reference_pth_sum(params, p), rtol=1e-5, atol=1e-6)


def test_missing_and_empty_gradients_are_ignored():
    # ``norm(ones(n))`` is ``sqrt(n)``, so the p=2 sum is the number of live elements.
    params = _params_from_specs([(3, 1.0), (0, 1.0), (2, 1.0), (7, None)])

    assert plugin.musa_local_pth_sum(params, 2.0).item() == pytest.approx(5.0)
    assert plugin.musa_local_max(params).item() == pytest.approx(1.0)

    assert plugin.musa_local_pth_sum([], 2.0).item() == 0.0
    assert plugin.musa_local_max([]).item() == 0.0


def test_all_gradients_missing_is_zero():
    params = _params_from_specs([(3, None), (5, None)])

    assert plugin.musa_local_pth_sum(params, 2.0).item() == 0.0
    assert plugin.musa_local_max(params).item() == 0.0


def test_zero_element_gradient_is_kept_off_the_batched_path():
    """An empty gradient must not enter the batched list.

    ``_foreach_norm`` raises on an empty input once the list spans two devices,
    so such a gradient keeps the per-tensor spelling -- which is what makes the
    plugin agree with the shared helpers here (on MUSA ``abs().max()`` of an empty
    tensor is 0.0; on the host it raises, exactly as the shared code does).
    """
    params = _params_from_specs([(0, 1.0), (2, 1.0)])

    assert plugin.musa_local_max(params).item() == pytest.approx(SHARED_MAX(params).item())
    assert plugin.musa_local_pth_sum(params, 2.0).item() == pytest.approx(SHARED_PTH_SUM(params, 2.0).item())


def test_apply_is_idempotent_and_revert_restores_the_shared_helpers():
    originals = (clip_grad_norm_module._local_pth_sum, clip_grad_norm_module._local_max)
    was_patched = plugin._PATCHED
    try:
        plugin._PATCHED = False
        plugin.apply_musa_fsdp2_clip_grad_norm_patch()
        assert clip_grad_norm_module._local_pth_sum is plugin.musa_local_pth_sum
        assert clip_grad_norm_module._local_max is plugin.musa_local_max

        first_pth_sum = clip_grad_norm_module._local_pth_sum
        plugin.apply_musa_fsdp2_clip_grad_norm_patch()
        assert clip_grad_norm_module._local_pth_sum is first_pth_sum

        plugin.revert_musa_fsdp2_clip_grad_norm_patch()
        assert clip_grad_norm_module._local_pth_sum is originals[0]
        assert clip_grad_norm_module._local_max is originals[1]

        # Reverting twice is a no-op rather than an error.
        plugin.revert_musa_fsdp2_clip_grad_norm_patch()
        assert clip_grad_norm_module._local_pth_sum is originals[0]
    finally:
        clip_grad_norm_module._local_pth_sum, clip_grad_norm_module._local_max = originals
        plugin._PATCHED = was_patched


def test_shared_helpers_were_captured_unpatched():
    """Guard the differential tests above, which compare against ``SHARED_*``.

    If the plugin had already rebound the shared module when this file was
    imported, those comparisons would silently compare the plugin with itself.
    """
    assert SHARED_PTH_SUM is not plugin.musa_local_pth_sum
    assert SHARED_MAX is not plugin.musa_local_max


def test_norms_are_gathered_onto_the_requested_device():
    """The contract ``_batched_norms`` has to keep for ``torch.stack`` to work.

    ``meta`` stands in for a second device type so this runs without an
    accelerator: dropping the ``off_device`` gathering would leave the norms on
    their original device and fail here.
    """
    params = _params_from_specs([(3, 1.0), (4, 1.0)], device="cpu")

    norms = plugin._batched_norms(params, ord=2.0, device=torch.device("meta"))

    assert len(norms) == len(params)
    assert {n.device.type for n in norms} == {"meta"}


@pytest.mark.skipif(DEVICE == "cpu", reason="needs an accelerator to pair with host gradients")
def test_mixed_host_and_accelerator_gradients_are_supported():
    """The cpu-offload path can mix host-resident and device-resident gradients.

    ``torch.stack`` rejects such a list, so the norms have to be gathered onto the
    reduce device before stacking.
    """
    params = _params_from_specs([(3, 1.0)], device="cpu") + _params_from_specs([(3, 1.0)])

    actual = plugin.musa_local_pth_sum(params, 2.0)

    assert actual.device.type == DEVICE
    assert actual.item() == pytest.approx(6.0)


@pytest.mark.skipif(DEVICE == "cpu", reason="needs an accelerator to pair with host gradients")
def test_mixed_devices_agree_with_the_shared_implementation():
    params = _params_from_specs([(3, 1.0), (4, 0.5)], device="cpu") + _params_from_specs([(2, 1.5)])

    assert torch.equal(plugin.musa_local_max(params).cpu(), SHARED_MAX(params).cpu())
    torch.testing.assert_close(
        plugin.musa_local_pth_sum(params, 2.0).cpu(), SHARED_PTH_SUM(params, 2.0).cpu(), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("norm_type", [2.0, float("inf")])
def test_patch_reaches_the_real_reduce_group_call_site(norm_type):
    """``_fsdp2_reduce_group`` looks the helpers up by name, so the rebinding must take effect.

    ``reduce_groups=[]`` keeps this free of any collective, which makes the two
    clip entry points (extra-parallel and cpu-offload) share exactly this path.
    """
    params = _random_params(dtypes=(torch.float16, torch.bfloat16, torch.float32))

    expected = clip_grad_norm_module._fsdp2_reduce_group(params, norm_type, [])
    with patched():
        actual = clip_grad_norm_module._fsdp2_reduce_group(params, norm_type, [])
        # The rebound helper really is the one the call site resolved.
        assert clip_grad_norm_module._fsdp2_reduce_group.__globals__["_local_pth_sum"] is not SHARED_PTH_SUM
        assert clip_grad_norm_module._fsdp2_reduce_group.__globals__["_local_max"] is not SHARED_MAX

    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-5, atol=1e-6)

    # ... and after reverting, the shared helpers are back in place.
    reverted = clip_grad_norm_module._fsdp2_reduce_group(params, norm_type, [])
    assert torch.equal(reverted.cpu(), expected.cpu())


@pytest.mark.parametrize("p", [1.0, 2.0])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_zero_element_gradient_does_not_raise_in_either_helper(dtype, p):
    """A zero-element gradient must behave exactly as it does for the shared helpers.

    The batched reduction returns 0 for an empty input, and the fallback for wide
    dtypes has to spell the max as ``abs().max()`` -- ``vector_norm(ord=inf)``
    raises there, which the shared max path never did.

    ``_local_pth_sum`` is only compared at finite ``p``: the reduce group routes
    ``norm_type=inf`` to ``_local_max``, so its infinity behaviour is unreachable.
    """
    params = _params_from_specs([(0, 1.0), (2, 1.0)], dtype=dtype)

    assert plugin.musa_local_max(params).item() == pytest.approx(SHARED_MAX(params).item())
    torch.testing.assert_close(
        plugin.musa_local_pth_sum(params, p).cpu(), SHARED_PTH_SUM(params, p).cpu(), rtol=1e-5, atol=1e-6
    )


def test_detached_local_grad_unwraps_a_dtensor_shard():
    """FSDP2 hands the clip path DTensor gradients; the helper must take the local shard."""
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    if not dist.is_initialized():
        # A file store keeps this free of ports and rendezvous environment variables.
        with tempfile.TemporaryDirectory() as tmp:
            dist.init_process_group("gloo", init_method=f"file://{tmp}/store", rank=0, world_size=1)
            _assert_dtensor_grad_is_unwrapped()
        return
    _assert_dtensor_grad_is_unwrapped()


def _assert_dtensor_grad_is_unwrapped():
    try:
        mesh = DeviceMesh("cpu", [0])
        local = torch.arange(6, dtype=torch.float32).reshape(2, 3).t().contiguous().t()
        param = torch.nn.Parameter(torch.zeros_like(local))
        param.grad = DTensor.from_local(local, mesh, [Shard(0)])

        detached = plugin._detached_local_grad(param)

        assert isinstance(detached, torch.Tensor)
        assert not isinstance(detached, DTensor)
        assert detached.device.type == "cpu"
        assert torch.equal(detached, local)
        # ... and the helpers accept such a gradient end to end.
        assert plugin.musa_local_pth_sum([param], 2.0).item() == pytest.approx(
            torch.linalg.vector_norm(local.to(torch.float32), 2.0).pow(2).item()
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_package_exports_the_installer_the_parallelize_path_imports():
    """``parallelize_model_fsdp2`` imports the installer from the platform package."""
    from veomni.ops.platform.musa import apply_musa_fsdp2_clip_grad_norm_patch

    assert apply_musa_fsdp2_clip_grad_norm_patch is plugin.apply_musa_fsdp2_clip_grad_norm_patch
