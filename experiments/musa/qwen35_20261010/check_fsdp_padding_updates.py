"""Paired FSDP2 gradients and AdamW updates for the exact odd shard size."""
from copy import deepcopy
import json
import os
from pathlib import Path
import torch
import torch_musa
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.utils.checkpoint import checkpoint
from veomni.ops.platform.musa.mccl_premul_sum import apply_mccl_premul_sum_patch
import install_mccl_padding as padding

rank = int(os.environ['RANK'])
torch.musa.set_device(int(os.environ['LOCAL_RANK']))
dist.init_process_group('mccl', device_id=torch.device('musa', int(os.environ['LOCAL_RANK'])))
apply_mccl_premul_sum_patch()
mesh = init_device_mesh('musa', (8,))
torch.manual_seed(881)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(1024, 14880, bias=False)
        self.extra = torch.nn.Parameter(torch.randn(2384) * .01)
        self.checkpoint = False

    def compute(self, x):
        y = self.proj(x)
        return torch.cat((y[:, :2384] + self.extra, y[:, 2384:]), dim=1)

    def forward(self, x):
        return checkpoint(self.compute, x, use_reentrant=False) if self.checkpoint else self.compute(x)


a = Model().to('musa')
b = deepcopy(a)
assert sum(p.numel() for p in a.parameters()) == 15239504
for model in (a, b):
    fully_shard(model, mesh=mesh, reshard_after_forward=True,
                mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32))
optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-4, foreach=False) for m in (a, b)]
padding.install()
report = []
for checkpoint_enabled in (False, True):
    a.checkpoint = b.checkpoint = checkpoint_enabled
    for step in range(3):
        torch.manual_seed(940 + rank * 10 + step)
        x = torch.randn(4, 1024, device='musa')
        losses = []
        grads = []
        for active, model, optimizer in zip((False, True), (a, b), optimizers):
            padding.enabled = active
            optimizer.zero_grad(set_to_none=True)
            loss = model(x).float().square().mean()
            loss.backward()
            torch.musa.synchronize()
            losses.append(loss.detach())
            grads.append({n: p.grad.to_local().clone() for n, p in model.named_parameters()})
        torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
        for name in grads[0]:
            torch.testing.assert_close(grads[0][name], grads[1][name], rtol=0, atol=0)
        for optimizer in optimizers:
            optimizer.step()
        torch.musa.synchronize()
        for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
            torch.testing.assert_close(pa.to_local(), pb.to_local(), rtol=0, atol=0)
        report.append({'checkpoint': checkpoint_enabled, 'step': step, 'loss': losses[0].item(),
                       'gradients_bitwise': True, 'parameters_bitwise': True})
padding.enabled = True
assert padding.calls['all_gather'] > 0 and padding.calls['reduce_scatter'] > 0
if rank == 0:
    path = Path('/data/share/liang.geng/fsdp_overlap_test/experiments/qwen35_optimize_20261009/fsdp_padding_updates.json')
    path.write_text(json.dumps({'results': report, 'stats': padding.stats()}, indent=2) + '\n')
    print('paired_updates', len(report), 'loss/gradients/parameters bitwise', padding.stats(), flush=True)
dist.destroy_process_group()
