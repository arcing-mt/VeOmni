"""Process-local prototype: reuse immutable host sequence metadata.

Weak references bound cache lifetime to the original tensor. Its PyTorch
version counter invalidates in-place updates, including normal view writes.
Inference tensors without version counters bypass caching. No installed file
is modified. Original numerical kernels and validation still run.
"""
import weakref

cache = {}
hits = 0
misses = 0


def host_cu(cu):
    global hits, misses
    if cu is None or cu.device.type == 'cpu':
        return cu
    try:
        version = cu._version
    except RuntimeError:
        misses += 1
        return cu.detach().cpu()
    key = id(cu)
    entry = cache.get(key)
    if entry is not None and entry[0]() is cu and entry[1] == version:
        hits += 1
        return entry[2]
    misses += 1
    value = cu.detach().cpu()
    def discard(ref):
        entry = cache.get(key)
        if entry is not None and entry[0] is ref:
            cache.pop(key, None)
    cache[key] = (weakref.ref(cu, discard), version, value)
    return value


def install():
    from veomni.ops.dispatch import OpSlot
    original_bind = OpSlot.bind
    def bind(self, implementation):
        result = original_bind(self, implementation)
        if self._kernel is None:
            return result
        if self.op_name == 'chunk_gated_delta_rule' and implementation == 'musa_tilelang':
            kernel = self._kernel
            def gdn(*args, **kwargs):
                cu = kwargs.get('cu_seqlens')
                if cu is not None:
                    kwargs['cu_seqlens'] = host_cu(cu)
                return kernel(*args, **kwargs)
            self._kernel = gdn
        elif self.op_name == 'causal_conv1d' and implementation == 'fla':
            kernel = self._kernel
            def conv(*args, **kwargs):
                cu = kwargs.get('cu_seqlens')
                if cu is not None and kwargs.get('cu_seqlens_cpu') is None:
                    kwargs['cu_seqlens_cpu'] = host_cu(cu)
                return kernel(*args, **kwargs)
            self._kernel = conv
        return result
    OpSlot.bind = bind


def stats():
    return {'hits':hits, 'misses':misses, 'live_entries':len(cache)}
