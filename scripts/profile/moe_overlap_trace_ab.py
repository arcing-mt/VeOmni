"""Compare two torch-profiler Chrome traces of the same training step.

The MoE overlap question ("is the shared-expert overlap worth it?") cannot be
answered from kernel totals or from end-to-end step time alone.  On this stack a
broken overlap shows up as:

  * cross-stream overlap going *down*, not up (the shared expert competes with
    DeepEP for SMs instead of hiding under it), and
  * extra GPU idle time, because the compute stream drains while a side stream
    fills.

So this script reports, per trace:

  * per-iteration span, GPU-busy union, and idle time
  * cross-stream overlap (sum of per-stream busy minus the union) and
    compute<->communication overlap, where "communication" is decided by kernel
    name (MCCL / DeepEP), not by stream id
  * idle gaps grouped into buckets, so a systematic per-layer stall is visible
  * kernel time by functional bucket, and the largest ON-OFF kernel deltas

Iteration boundaries come from the optimizer step (``musa::_fused_adamw_``),
which every supported configuration emits once per step; the window is the gap
between two consecutive optimizer steps, i.e. exactly one full step.

Usage:
    python scripts/profile/moe_overlap_trace_ab.py ON_TRACE OFF_TRACE
    python scripts/profile/moe_overlap_trace_ab.py A B --step 2   # pick a window
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict


GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")

#: Kernel-name patterns, most specific first.
BUCKETS: tuple[tuple[str, str], ...] = (
    ("ace_comm", r"deep_ep|deepep|intranode_ace|notify_dispatch|workspace_scan"),
    ("mccl", r"mccl|SendRecv|ReduceScatter|AllGather|AllReduce|AlltoAll"),
    ("grouped_gemm", r"ssgemm|group_gemm|gemm_gm|grouped_gemm"),
    ("gemm", r"gemm|matmul|conv3d"),
    ("attention", r"flash|attn|fmha"),
    ("copy", r"copy|cast|fill|IndexSelect|IndexFuncs|Concat"),
    ("reduce_sort", r"Reduce|reduce_kernel|Scan|scan|Norm|norm|bincount|sort|Sort|shuffle|topk|TopK|argsort|nonzero"),
    ("elementwise", r"kernel_unary|kernel_binary|silu|sigmoid|multi_tensor_apply"),
)

#: Buckets that carry token/expert communication rather than computation.
COMM_BUCKETS = frozenset({"ace_comm", "mccl"})

STEP_MARKER = "musa::_fused_adamw_"


def load(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as handle:
        return json.load(handle)["traceEvents"]


def merge(spans):
    """Merge overlapping [start, end) intervals into a disjoint sorted list."""
    if not spans:
        return []
    spans = sorted(spans)
    out = [list(spans[0])]
    for start, end in spans[1:]:
        if start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return [(a, b) for a, b in out]


def _length(spans) -> float:
    return sum(b - a for a, b in merge(spans))


def intersection(a, b) -> float:
    """Total time covered by both disjoint, sorted interval lists."""
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def bucket_of(name: str) -> str:
    for bucket, pattern in BUCKETS:
        if re.search(pattern, name, re.I):
            return bucket
    return "other"


#: Idle-gap histogram edges, largest first, so the first match wins.  Gaps
#: below the last edge are scheduler noise and are not counted.
GAP_BUCKETS: tuple[tuple[float, str], ...] = (
    (20_000, ">20ms"),
    (5_000, "5-20ms"),
    (2_000, "2-5ms"),
    (1_000, "1-2ms"),
    (500, "0.5-1ms"),
)


def step_boundaries(events: list[dict]) -> list[float]:
    """Start timestamps of the training steps.

    One training step runs the optimizer once, but a single optimizer step emits
    several kernels; ``step_boundaries`` groups launches that are close together
    relative to the observed spacing and returns the first timestamp of each
    group.  Deriving the cut from the data (rather than a fixed constant) keeps
    this working both for short-sequence debug configs, where steps are only a
    few hundred ms apart, and for real runs.

    CPU events carry the profiled process id while GPU kernels carry the
    device's, so this deliberately does not filter on ``pid``.
    """
    stamps = sorted(
        e["ts"] for e in events if e.get("cat") == "cpu_op" and e.get("name") == STEP_MARKER and e.get("ph") == "X"
    )
    if not stamps:
        raise SystemExit(
            f"no {STEP_MARKER!r} events in this trace; cannot locate step boundaries "
            "(re-profile with the default optimizer, or extend STEP_MARKER)"
        )
    if len(stamps) < 2:
        raise SystemExit("only one optimizer launch in this trace; profile at least two consecutive steps")

    gaps = sorted(stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1))
    median = gaps[len(gaps) // 2]
    total = stamps[-1] - stamps[0]
    if median * 5 >= total:
        # Launches are evenly spaced: there is no burst structure to exploit, so
        # treat every launch as its own step.  This is what a debug config with
        # one optimizer kernel per step looks like.
        return stamps

    # Intra-step gaps are launch spacing (tens of ms); inter-step gaps are a whole
    # step.  Anything well above the median is a step boundary.
    cut = max(10.0 * median, median + 1_000.0)
    bounds = [stamps[0]]
    for ts in stamps[1:]:
        if ts - bounds[-1] > cut:
            bounds.append(ts)
    if len(bounds) < 2:
        # A short profiling window can contain only one optimizer step.  Fall back
        # to the profiled kernel range so the script still reports a comparison,
        # but say so: that window is the whole trace, not exactly one step.
        return _single_step_fallback(events, stamps, cut)
    return bounds


def _single_step_fallback(events: list[dict], stamps: list[float], cut: float) -> list[float]:
    """Span the whole profiled region when only one optimizer step is present."""
    kernel_ts = [e["ts"] for e in events if e.get("cat") in GPU_CATS and e.get("ph") == "X"]
    start = min(stamps[0], min(kernel_ts)) if kernel_ts else stamps[0]
    end = max(kernel_ts) if kernel_ts else stamps[-1]
    print(
        f"  warning: only one optimizer step detected (launch-gap cut={cut:.0f} us); "
        "falling back to the whole profiled region, which may span more than one step",
        flush=True,
    )
    return [start, end]


def gpu_events(events: list[dict], pid) -> list[dict]:
    """Kernels of a single device process, so multi-device traces stay separate."""
    return [e for e in events if e.get("cat") in GPU_CATS and e.get("ph") == "X" and e.get("pid") == pid]


def pick_pid(events: list[dict]):
    counts: dict = defaultdict(int)
    for e in events:
        if e.get("cat") in GPU_CATS and e.get("ph") == "X":
            counts[e.get("pid")] += 1
    if not counts:
        raise SystemExit("trace contains no GPU kernel events")
    return max(counts, key=counts.get)


def analyze(label: str, path: str, step: int) -> dict:
    events = load(path)
    pid = pick_pid(events)
    bounds = step_boundaries(events)
    if not 0 <= step < len(bounds) - 1:
        raise SystemExit(f"{label}: --step {step} out of range (trace has {len(bounds) - 1} steps)")
    lo, hi = bounds[step], bounds[step + 1]
    span = hi - lo

    kernels = [e for e in gpu_events(events, pid) if lo <= e["ts"] < hi]
    # Clip to the window so `busy` can never exceed `span`.
    spans = [(max(e["ts"], lo), min(e["ts"] + e.get("dur", 0.0), hi)) for e in kernels]
    union = merge(spans)
    busy = _length(union)

    hist: dict[str, int] = defaultdict(int)
    for i in range(len(union) - 1):
        gap = union[i + 1][0] - union[i][1]
        for limit, name in GAP_BUCKETS:
            if gap > limit:
                hist[name] += 1
                break

    per_stream: dict = defaultdict(list)
    per_bucket: dict = defaultdict(list)
    per_bucket_time: dict = defaultdict(float)
    per_kernel: dict = defaultdict(float)
    for e, clipped in zip(kernels, spans):
        stream = e.get("tid", -1)
        bucket = bucket_of(e["name"])
        per_stream[stream].append(clipped)
        per_bucket["comm" if bucket in COMM_BUCKETS else "rest"].append(clipped)
        per_bucket_time[bucket] += clipped[1] - clipped[0]
        per_kernel[e["name"]] += clipped[1] - clipped[0]

    busy_sum = sum(_length(v) for v in per_stream.values())
    cross_overlap = busy_sum - busy
    comm_spans = merge(per_bucket["comm"])
    comm_overlap = intersection(merge(per_bucket["rest"]), comm_spans)

    print(f"\n{'=' * 88}\n{label}: {path}\n{'=' * 88}")
    print(f"  pid {pid}   window step {step}  [{lo:.0f}, {hi:.0f})")
    print(f"  iteration span        {span / 1000:10.1f} ms")
    print(f"  GPU busy (union)      {busy / 1000:10.1f} ms   ({100 * busy / span:.1f}% of span)")
    print(f"  GPU idle              {(span - busy) / 1000:10.1f} ms")
    print(f"  non-comm stream overlap {cross_overlap / 1000:8.1f} ms   ({100 * cross_overlap / span:.1f}% of span)")
    print(f"  compute<->comm overlap{comm_overlap / 1000:10.1f} ms   (comm = DeepEP + MCCL kernels)")
    if not comm_spans:
        # Without a comm match the two overlap figures above are meaningless: every
        # kernel is folded into "compute".  Say so instead of printing a number.
        print("  !! no DeepEP/MCCL kernel matched -- the overlap figures above are NOT valid.")
        print("     Largest unmatched kernels (extend BUCKETS if these are communication):")
        for name, dur in sorted(per_kernel.items(), key=lambda kv: -kv[1])[:5]:
            print(f"       {dur / 1000:9.1f} ms  {name[:80]}")
    print(f"  idle gaps >0.5ms      {dict(hist)}")
    print(f"  streams with work     {len(per_stream)}")
    print("  per-stream busy (top 6):")
    for tid, v in sorted(per_stream.items(), key=lambda kv: -_length(kv[1]))[:6]:
        print(f"      stream {tid:>5}: {_length(v) / 1000:9.1f} ms")
    print("  kernel time by bucket:")
    for name, dur in sorted(per_bucket_time.items(), key=lambda kv: -kv[1]):
        print(f"      {name:14s} {dur / 1000:9.1f} ms")

    return dict(
        span=span,
        busy=busy,
        buckets=per_bucket_time,
        kernels=per_kernel,
        cross_overlap=cross_overlap,
        comm_overlap=comm_overlap,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("on_trace", help="trace with moe_shared_expert_overlap=true")
    parser.add_argument("off_trace", help="trace with moe_shared_expert_overlap=false")
    parser.add_argument("--step", type=int, default=0, help="which step window to analyse (default: first)")
    parser.add_argument("--top", type=int, default=12, help="kernel deltas to print")
    args = parser.parse_args()

    on = analyze("OVERLAP ON", args.on_trace, args.step)
    off = analyze("OVERLAP OFF", args.off_trace, args.step)

    print(f"\n{'#' * 88}\n# DELTA (ON - OFF)\n{'#' * 88}")
    print(f"  span                   {on['span'] / 1000:+9.1f} ms")
    print(f"  busy                   {on['busy'] / 1000:+9.1f} ms")
    print(f"  cross-stream overlap   {on['cross_overlap'] / 1000:+9.1f} ms")
    print(f"  compute<->comm overlap {on['comm_overlap'] / 1000:+9.1f} ms")
    print("  bucket delta:")
    ordered = sorted(
        set(on["buckets"]) | set(off["buckets"]),
        key=lambda k: -(on["buckets"].get(k, 0) + off["buckets"].get(k, 0)),
    )
    for name in ordered:
        delta = (on["buckets"].get(name, 0.0) - off["buckets"].get(name, 0.0)) / 1000
        if abs(delta) > 0.5:
            print(f"      {name:14s} {delta:+9.1f} ms")

    print(f"  top {args.top} kernel deltas:")
    names = set(on["kernels"]) | set(off["kernels"])
    rows = sorted(((on["kernels"].get(n, 0.0) - off["kernels"].get(n, 0.0)) / 1000, n) for n in names)
    for delta, name in rows[::-1][: args.top]:
        print(f"      {delta:+9.1f} ms  {name[:92]}")


if __name__ == "__main__":
    main()
