"""Loss and full-gradient gates for ignored-token compaction."""
import json
from pathlib import Path
import statistics
import time
import torch
import torch_musa
from veomni.distributed.parallel_state import get_parallel_state
from veomni.ops.kernels.cross_entropy.chunk_loss import chunk_loss_function
from supervised_chunk_loss import compact_chunk_loss, stats

assert not get_parallel_state().sp_enabled


def error(a,b):
    delta=a.float()-b.float()
    return {'max_abs':delta.abs().max().item(),'rms_rel':(delta.square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-12)).item()}


def run(b,s,h,v,fraction,shift=False):
    torch.manual_seed(641)
    x=torch.randn(b,s,h,device='musa',dtype=torch.bfloat16).requires_grad_()
    w=(torch.randn(v,h,device='musa',dtype=torch.bfloat16)*.02).requires_grad_()
    labels=torch.randint(0,v,(b,s),device='musa')
    labels[torch.rand(b,s,device='musa')>=fraction]=-100
    kwargs={'shift_labels':labels} if shift else {}
    def operation(compact):
        x.grad=w.grad=None
        f=compact_chunk_loss if compact else chunk_loss_function
        out,_=f(x,w,labels,vocab_size=v,**kwargs)
        out.backward()
        return [out.detach(),x.grad,w.grad]
    ref=[a.clone() for a in operation(False)]
    actual=operation(True)
    errors=[error(a,b) for a,b in zip(actual,ref)]
    assert errors[0]['rms_rel']<.0001 and all(e['rms_rel']<.02 for e in errors[1:]),errors
    for _ in range(2):operation(False);operation(True)
    torch.musa.synchronize()
    samples={False:[],True:[]}
    for it in range(8):
        for c in ((False,True) if it%2==0 else (True,False)):
            start=time.perf_counter();operation(c);torch.musa.synchronize()
            samples[c].append((time.perf_counter()-start)*1000)
    d={'shape':[b,s,h,v],'fraction':fraction,'explicit_shift':shift,'errors':errors,
       'baseline_ms':statistics.median(samples[False]),'compact_ms':statistics.median(samples[True]),'samples':{str(k):v for k,v in samples.items()}}
    print(json.dumps(d),flush=True);return d


report=[run(2,128,128,257,.2),run(2,128,128,257,.2,True),run(1,8155,2048,248320,.2)]
Path('/data/share/liang.geng/fsdp_overlap_test/experiments/qwen35_optimize_20261009/supervised_loss_parity.json').write_text(json.dumps({'results':report,'stats':stats()},indent=2)+'\n')
