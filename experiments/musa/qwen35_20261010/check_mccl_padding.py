"""Full collective output gate on random data, both sync modes and streams."""
import json
import os
from pathlib import Path
import torch
import torch_musa
import torch.distributed as dist
from torch.distributed.fsdp._fully_shard._fsdp_collectives import DefaultAllGather, DefaultReduceScatter
import install_mccl_padding as padding

rank = int(os.environ['RANK'])
torch.musa.set_device(int(os.environ['LOCAL_RANK']))
dist.init_process_group('mccl', device_id=torch.device('musa', int(os.environ['LOCAL_RANK'])))
torch.manual_seed(701 + rank)
padding.install()
report = []
for dtype in (torch.bfloat16, torch.float32):
    for n in (1904938, 1904944, 13):
        for async_op in (False, True):
            for separate_stream in (False, True):
                stream = torch.musa.Stream() if separate_stream else torch.musa.current_stream()
                with torch.musa.stream(stream):
                    for kind in ('AG', 'RS'):
                        src = torch.randn(n if kind == 'AG' else n * 8, device='musa', dtype=dtype)
                        dst = torch.empty(n * 8 if kind == 'AG' else n, device='musa', dtype=dtype)
                        collective = DefaultAllGather() if kind == 'AG' else DefaultReduceScatter()
                        def run(active):
                            padding.enabled = active
                            work = (collective(dst, src, dist.group.WORLD, async_op=async_op) if kind == 'AG'
                                    else collective(dst, src, dist.group.WORLD, dist.ReduceOp.SUM, async_op=async_op))
                            if work is not None:
                                work.wait()
                            return dst.clone()
                        reference = run(False)
                        actual = run(True)
                        stream.synchronize()
                        if kind == 'AG':
                            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                        else:
                            torch.testing.assert_close(actual, reference, rtol=.02 if dtype == torch.bfloat16 else 1e-5,
                                                       atol=.125 if dtype == torch.bfloat16 else 2e-6)
                        delta = actual.float() - reference.float()
                        relative = (delta.square().mean().sqrt() / reference.float().square().mean().sqrt()).item()
                        assert relative < (.005 if dtype == torch.bfloat16 else 1e-6), relative
                        report.append({'op': kind, 'dtype': str(dtype), 'n': n,
                                       'async': async_op, 'separate_stream': separate_stream,
                                       'bitwise': torch.equal(actual, reference), 'rms_relative': relative})
                        del src, dst, reference, actual
                dist.barrier()
padding.enabled = True
if rank == 0:
    out = Path('/data/share/liang.geng/fsdp_overlap_test/experiments/qwen35_optimize_20261009/mccl_padding_correctness.json')
    out.write_text(json.dumps({'checks_per_rank': len(report), 'results': report, 'stats': padding.stats()}, indent=2) + '\n')
    print('checks_per_rank', len(report), 'max_rms', max(x['rms_relative'] for x in report), 'stats', padding.stats(), flush=True)
dist.destroy_process_group()
