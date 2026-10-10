"""Non-default stream cache reuse and live-autograd LRU eviction gate."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_musa
from veomni.models.transformers.qwen3_5_moe.generated import patched_modeling_qwen3_5_moe_gpu as modeling
from install_vision_embedding_csr import make_candidate
import vision_embedding_csr as artifact

torch.musa.set_device(0)
torch.manual_seed(38195)
table = torch.nn.Embedding(2304, 1152, dtype=torch.bfloat16, device='musa')
model = SimpleNamespace(pos_embed=table, num_grid_per_side=48, spatial_merge_size=2,
                        device=table.weight.device)
native = modeling.Qwen3_5MoeVisionModel.fast_pos_embed_interpolate
candidate = make_candidate(native)
grids = [(1,26,40), (1,40,26), (2,40,26)]
upstreams = [torch.randn(t*h*w,1152,device='musa',dtype=torch.bfloat16)*.1 for t,h,w in grids]
expected = []
for grid, gradient in zip(grids, upstreams):
    output = native(model, [grid])
    dw, = torch.autograd.grad(output, table.weight, gradient)
    expected.append((output.detach(),dw))

first, second = torch.musa.Stream(), torch.musa.Stream()
initialized = torch.musa.Event()
initialized.record(torch.musa.current_stream())
first.wait_event(initialized)
second.wait_event(initialized)
saved_limit = artifact.MAX_CACHED_GRIDS
artifact.MAX_CACHED_GRIDS = 1
with torch.musa.stream(first):
    out_a = candidate(model, [grids[0]])
    first_ready = torch.musa.Event()
    first_ready.record()
with torch.musa.stream(second):
    # Evicts A's metadata while out_a's autograd context remains live.
    out_b = candidate(model, [grids[1]])
    second_ready = torch.musa.Event()
    second_ready.record()
with torch.musa.stream(first):
    # The same B key now hits metadata originally produced on second stream.
    out_c = candidate(model, [grids[2]])
    third_ready = torch.musa.Event()
    third_ready.record()
consumer = torch.musa.current_stream()
consumer.wait_event(first_ready)
consumer.wait_event(second_ready)
consumer.wait_event(third_ready)
observed = []
for output, gradient in zip((out_a,out_b,out_c),upstreams):
    dw, = torch.autograd.grad(output,table.weight,gradient)
    # Autograd's documented stream waits expose these to the calling stream.
    observed.append((output.detach().clone(),dw.clone()))
torch.musa.synchronize()
for actual, reference in zip(observed,expected):
    assert all(torch.equal(x,y) for x,y in zip(actual,reference))
assert artifact.counters == {'calls':3,'cache_misses':2,'evictions':1,'cache_hits':1}, artifact.stats()
assert sum(map(len,artifact._cache.values())) == 1
artifact.MAX_CACHED_GRIDS = saved_limit
report = {'status':'passed','scope':'Two non-default FW streams, cross-stream same-key hit, '
          'LRU limit1 eviction while first autograd context live, per-call native FW/DW bitwise; '
          'no claim that pending event query or random arbitrary multi-stream gradient accumulation was tested.',
          'cases':[list(g) for g in grids], 'stats':artifact.stats()}
Path(os.environ['VISION_EMBEDDING_OUTPUT']).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
