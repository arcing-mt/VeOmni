"""Check TE accumulation against CPU double, including new trial guards."""
import os
import torch
import torch_musa
os.environ['VEOMNI_MUSA_DEEPEP_COUNTING_SORT'] = '1'
from veomni.distributed.moe import deepep_ace
import install_te_compact
install_te_compact.install()
torch.manual_seed(32)
n,h,e,k = 257,128,32,8
indices = torch.rand(n,e).topk(k,dim=-1).indices
indices[torch.rand(n,k)>.2] = -1
slots = (indices.reshape(-1)>=0).nonzero().flatten()
slots = slots[torch.argsort(indices.reshape(-1)[slots],stable=True)]
rows = slots // k
counts = torch.bincount(indices[indices>=0],minlength=e).tolist()
x = torch.randn(n,h,device='musa',dtype=torch.bfloat16).requires_grad_()
p = torch.rand(n,k,device='musa').requires_grad_()
y,pr,metadata,_ = deepep_ace._compact_permute(x,indices.musa(),p,e,counts)
dy = torch.randn_like(y)
dpr = torch.randn_like(pr)
torch.autograd.backward((y,pr),(dy,dpr))
gold_dx = torch.zeros(n,h,dtype=torch.float64).index_add_(0,rows,dy.cpu().double()).bfloat16()
assert torch.equal(x.grad.cpu(),gold_dx)
gold_dp = torch.zeros(n*k).index_add_(0,slots,dpr.cpu()).reshape(n,k)
assert torch.equal(p.grad.cpu(),gold_dp)

y = y.detach().requires_grad_();pr = pr.detach().requires_grad_()
out = deepep_ace._compact_unpermute(y,pr,metadata,n)
weighted = (y.detach()*pr.detach().to(y.dtype).unsqueeze(-1)).cpu()
gold = torch.zeros(n,h,dtype=torch.float64).index_add_(0,rows,weighted.double()).bfloat16()
assert torch.equal(out.cpu(),gold)
out.backward(torch.randn_like(out))
with torch.autocast('musa',dtype=torch.bfloat16):
    assert torch.is_autocast_enabled('musa')
    a = torch.randn(32,64,device='musa')
    b = torch.randn(64,32,device='musa')
    assert (a@b).dtype == torch.bfloat16

empty_ids = torch.full((n,k),-1,device='musa',dtype=torch.long)
empty = deepep_ace._compact_permute(x,empty_ids,p,e,[0]*e)
assert empty[0].shape[0] == 0 and not hasattr(empty[2],'_te_compaction')
print('CPU double accumulation, gradients, zero-route fallback and real autocast matmul: PASS',flush=True)
