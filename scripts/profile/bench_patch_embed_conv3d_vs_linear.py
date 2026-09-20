"""Microbenchmark: ViT patch-embed ``Conv3d`` vs the equivalent ``Linear`` (unfold + GEMM).

The Qwen3.5-MoE ViT patch embedder is

    Conv3d(in_channels=3, out_channels=1152, kernel_size=(2,16,16), stride=(2,16,16), padding=0)

called on ``hidden_states.view(N, 3, 2, 16, 16)``.  Because ``kernel_size == stride`` and
``padding == 0`` the output has spatial size 1x1x1, so the convolution degenerates into a single
matrix product over the flattened kernel window:

    out[n, m] = sum_{c,t,h,w} w[m, c, t, h, w] * x[n, c, t, h, w] + b[m]
              = Linear(x.view(N, 1536), w.view(1152, 1536), b)

Both spellings are therefore mathematically identical (they differ only in the order in which the
1536-term reduction is accumulated).  This script

  1. checks the numerical agreement of the forward output and of the weight gradient, and
  2. measures the per-call device time of both spellings on the shapes the trainer actually uses.

``pixel_values`` is a leaf input that never requires grad, so the backward pass of this op is a
weight-gradient GEMM only -- there is no dgrad.

Usage::

    MUSA_VISIBLE_DEVICES=0 python scripts/profile/bench_patch_embed_conv3d_vs_linear.py
    MUSA_VISIBLE_DEVICES=0 python scripts/profile/bench_patch_embed_conv3d_vs_linear.py --ns 1008,18048
"""

import argparse
import gzip
import json
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F


# Qwen3.5-35B-A3B vision_config: hidden_size=1152, in_channels=3, patch_size=16, temporal_patch_size=2.
DEFAULT_M = 1152
DEFAULT_C = 3
DEFAULT_T = 2
DEFAULT_P = 16
# The flatten order the fold relies on: (c, t, h, w) of the Conv3d window.
WINDOW = DEFAULT_C * DEFAULT_T * DEFAULT_P * DEFAULT_P


def resolve_device(device: str) -> torch.device:
    dev = torch.device(device)
    if dev.type == "musa" and not torch.musa.is_available():
        raise SystemExit("MUSA is not available")
    return dev


def device_module(dev: torch.device):
    return torch.musa if dev.type == "musa" else torch.cuda


def sync(dev: torch.device) -> None:
    device_module(dev).synchronize()


def make_tensors(n: int, m: int, k: int, dev: torch.device):
    """Image-like activations (U[-1,1]) and ``initializer_range``-like weights."""
    gen = torch.Generator(device="cpu").manual_seed(0)
    x = (torch.rand(n, k, generator=gen, dtype=torch.float32) * 2.0 - 1.0).to(dev)
    w = (torch.randn(m, k, generator=gen, dtype=torch.float32) * 0.02).to(dev)
    b = torch.zeros(m, dtype=torch.float32, device=dev)
    return x, w.view(m, DEFAULT_C, DEFAULT_T, DEFAULT_P, DEFAULT_P), b


def conv_module(m: int, dev: torch.device, dtype: torch.dtype) -> torch.nn.Conv3d:
    return torch.nn.Conv3d(
        DEFAULT_C,
        m,
        kernel_size=(DEFAULT_T, DEFAULT_P, DEFAULT_P),
        stride=(DEFAULT_T, DEFAULT_P, DEFAULT_P),
        bias=True,
    ).to(device=dev, dtype=dtype)


def stats(name: str, ref: torch.Tensor, got: torch.Tensor) -> dict:
    # MUSA has essentially no fp64 support, so the upcast and the statistics run on the CPU.
    ref64, got64 = ref.detach().cpu().double(), got.detach().cpu().double()
    diff = (got64 - ref64).abs()
    return {
        "name": name,
        "max_abs": diff.max().item(),
        "rms_rel": ((diff**2).mean().sqrt() / ref64.pow(2).mean().sqrt()).item(),
        "cos": F.cosine_similarity(got64.flatten(), ref64.flatten(), dim=0).item(),
        "ref_absmax": ref64.abs().max().item(),
    }


# --------------------------------------------------------------------------------------
# 1. numerical alignment
# --------------------------------------------------------------------------------------
def correctness(n: int, m: int, k: int, dev: torch.device) -> None:
    print(f"\n=== numerical alignment (N={n}, M={m}, K={k}) ===")
    xf, wf, bf = make_tensors(n, m, k, dev)
    x5f = xf.view(n, DEFAULT_C, DEFAULT_T, DEFAULT_P, DEFAULT_P)
    conv32 = conv_module(m, dev, torch.float32)
    with torch.no_grad():
        conv32.weight.copy_(wf)
        conv32.bias.copy_(bf)

    x = xf.to(torch.bfloat16)
    x5 = x.view(n, DEFAULT_C, DEFAULT_T, DEFAULT_P, DEFAULT_P)
    conv16 = conv_module(m, dev, torch.bfloat16)
    with torch.no_grad():
        conv16.weight.copy_(wf.to(torch.bfloat16))
        conv16.bias.copy_(wf.new_zeros(m).to(torch.bfloat16))

    # fp64 ground truth.  muDNN has no fp64 GEMM, so the reference is computed on the CPU.
    # The reference must use exactly the same (quantized) inputs the kernels see.
    def ref64(xx, ww, bb):
        return xx.detach().cpu().double() @ ww.detach().cpu().double().view(m, k).T + bb.detach().cpu().double()

    r32 = ref64(xf, conv32.weight, conv32.bias)
    r16 = ref64(x, conv16.weight, conv16.bias)

    rows = []
    with torch.no_grad():
        conv_fp32 = conv32(x5f).view(n, m)
        lin_fp32 = F.linear(xf, conv32.weight.view(m, k), conv32.bias)
        rows.append(stats("conv3d fp32", r32, conv_fp32))
        rows.append(stats("linear fp32", r32, lin_fp32))
        rows.append(stats("linear fp32 vs conv3d fp32", conv_fp32, lin_fp32))
        conv_bf16 = conv16(x5).view(n, m)
        lin_bf16 = F.linear(x, conv16.weight.view(m, k), conv16.bias)
        rows.append(stats("conv3d bf16", r16, conv_bf16))
        rows.append(stats("linear bf16", r16, lin_bf16))
        rows.append(stats("linear bf16 vs conv3d bf16", conv_bf16, lin_bf16))

    # weight gradient (the only gradient this op produces in the model).
    gout = (torch.randn(n, m, dtype=torch.float32, device=dev) * 0.02).to(torch.bfloat16)
    g16_ref = gout.detach().cpu().double().T @ x.detach().cpu().double()
    g_conv = torch.autograd.grad(conv16(x5).view(n, m), conv16.weight, gout)[0].view(m, k)
    g_lin = torch.autograd.grad(F.linear(x, conv16.weight.view(m, k), conv16.bias), conv16.weight, gout)[0].view(m, k)
    rows.append(stats("wgrad conv3d bf16", g16_ref, g_conv))
    rows.append(stats("wgrad linear bf16", g16_ref, g_lin))
    rows.append(stats("wgrad linear vs conv3d bf16", g_conv, g_lin))

    gout32 = torch.randn(n, m, dtype=torch.float32, device=dev) * 0.02
    g32_ref = gout32.detach().cpu().double().T @ xf.detach().cpu().double()
    g32_conv = torch.autograd.grad(conv32(x5f).view(n, m), conv32.weight, gout32)[0].view(m, k)
    g32_lin = torch.autograd.grad(F.linear(xf, conv32.weight.view(m, k), conv32.bias), conv32.weight, gout32)[0].view(
        m, k
    )
    rows.append(stats("wgrad conv3d fp32", g32_ref, g32_conv))
    rows.append(stats("wgrad linear fp32", g32_ref, g32_lin))

    print(f"{'variant':<32} {'ref |max|':>11} {'max_abs':>11} {'rms_rel':>11} {'cosine':>18}")
    for r in rows:
        print(
            f"{r['name']:<32} {r['ref_absmax']:>11.3e} {r['max_abs']:>11.3e} {r['rms_rel']:>11.3e} {r['cos']:>18.12f}"
        )


# --------------------------------------------------------------------------------------
# 2. performance
# --------------------------------------------------------------------------------------
def time_loop(fn, iters: int, warmup: int, dev: torch.device) -> float:
    for _ in range(warmup):
        fn()
    sync(dev)
    start = device_module(dev).Event(enable_timing=True)
    end = device_module(dev).Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    sync(dev)
    return start.elapsed_time(end) / iters


def bench(n: int, m: int, k: int, dev: torch.device, iters: int, warmup: int) -> dict:
    _, w, _ = make_tensors(n, m, k, dev)
    x = torch.rand(n, k, dtype=torch.bfloat16, device=dev) * 2 - 1
    x5 = x.view(n, DEFAULT_C, DEFAULT_T, DEFAULT_P, DEFAULT_P)
    conv = conv_module(m, dev, torch.bfloat16)
    with torch.no_grad():
        conv.weight.copy_(w.to(torch.bfloat16))
    wflat = conv.weight.view(m, k)
    gout = (torch.randn(n, m, dtype=torch.float32, device=dev) * 0.02).to(torch.bfloat16)
    gout5 = gout.view(n, m, 1, 1, 1)

    def conv_fwd():
        return conv(x5)

    def lin_fwd():
        return F.linear(x, wflat, conv.bias)

    def conv_step():
        conv.zero_grad(set_to_none=True)
        conv(x5).backward(gout5)

    def lin_step():
        conv.zero_grad(set_to_none=True)
        F.linear(x, wflat, conv.bias).backward(gout)

    with torch.no_grad():
        t_conv_fwd = time_loop(conv_fwd, iters, warmup, dev)
        t_lin_fwd = time_loop(lin_fwd, iters, warmup, dev)
    t_conv_step = time_loop(conv_step, iters, warmup, dev)
    t_lin_step = time_loop(lin_step, iters, warmup, dev)

    flops = 2.0 * n * m * k
    return {
        "n": n,
        "flops": flops,
        "conv_fwd": t_conv_fwd,
        "lin_fwd": t_lin_fwd,
        "conv_fwd_bwd": t_conv_step,
        "lin_fwd_bwd": t_lin_step,
        "conv_bwd": t_conv_step - t_conv_fwd,
        "lin_bwd": t_lin_step - t_lin_fwd,
    }


def profile_kernels(kind: str, n: int, m: int, k: int, dev: torch.device, iters: int = 5) -> None:
    from torch.profiler import ProfilerActivity, profile

    device_activity = getattr(ProfilerActivity, dev.type.upper(), ProfilerActivity.CUDA)

    _, w, _ = make_tensors(n, m, k, dev)
    x = torch.rand(n, k, dtype=torch.bfloat16, device=dev) * 2 - 1
    x5 = x.view(n, DEFAULT_C, DEFAULT_T, DEFAULT_P, DEFAULT_P)
    conv = conv_module(m, dev, torch.bfloat16)
    with torch.no_grad():
        conv.weight.copy_(w.to(torch.bfloat16))
    wflat = conv.weight.view(m, k)
    gout = (torch.randn(n, m, dtype=torch.float32, device=dev) * 0.02).to(torch.bfloat16)

    def step():
        conv.zero_grad(set_to_none=True)
        if kind == "conv3d":
            conv(x5).backward(gout.view(n, m, 1, 1, 1))
        else:
            F.linear(x, wflat, conv.bias).backward(gout)

    for _ in range(3):
        step()
    sync(dev)
    with profile(activities=[ProfilerActivity.CPU, device_activity]) as prof:
        for _ in range(iters):
            step()
    sync(dev)

    trace_path = f"/tmp/bench_patch_embed_{kind}_trace.json.gz"
    prof.export_chrome_trace(trace_path)
    with gzip.open(trace_path, "rt") as f:
        trace = json.load(f)
    agg = defaultdict(lambda: [0, 0.0])
    for ev in trace["traceEvents"]:
        if ev.get("cat") == "kernel":
            agg[ev["name"]][0] += 1
            agg[ev["name"]][1] += ev.get("dur", 0.0)
    print(f"\n  -- device kernels per fwd+bwd step ({kind}, N={n}) --")
    for name, (cnt, us) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:8]:
        print(f"    {us / 1000.0 / iters:8.3f} ms/step  n={cnt / iters:<5.1f}  {name[:105]}")


def probe_dtype_precision(dev: torch.device, m: int = 512, n: int = 2048, k: int = 1536) -> None:
    """Report how far MUSA's own GEMMs are from an fp64 reference, to set the noise floor."""
    gen = torch.Generator(device="cpu").manual_seed(0)
    x = torch.rand(n, k, generator=gen, dtype=torch.float32) * 2 - 1
    w = torch.randn(m, k, generator=gen, dtype=torch.float32) * 0.02
    ref = x.double() @ w.double().T
    print(f"\n=== MUSA GEMM precision probe ({n}x{k}x{m}) ===")
    for dt in (torch.float32, torch.bfloat16):
        got = (x.to(dev).to(dt) @ w.to(dev).to(dt).T).float().cpu().double()
        err = (got - ref).abs().max().item()
        # mantissa bits implied by the worst-case relative error
        span = ref.abs().max().item()
        bits = -torch.log2(torch.tensor(max(err / span, 1e-12))).item()
        print(f"  dtype={str(dt):<16} max_abs_err={err:.3e}  |ref|max={span:.3e}  ~{bits:.1f} mantissa bits")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ns", default="1008,18048", help="comma separated patch counts N to sweep")
    ap.add_argument("--m", type=int, default=DEFAULT_M, help="out channels / embed dim")
    ap.add_argument("--device", default="musa:0")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--profile", action="store_true", help="also dump per-kernel device time")
    ap.add_argument("--skip-correctness", action="store_true")
    args = ap.parse_args()

    dev = resolve_device(args.device)
    ns = [int(v) for v in args.ns.split(",") if v]
    print(f"device={dev} ({device_module(dev).get_device_name(dev)}), dtype=bfloat16, M={args.m}, K={WINDOW}")

    if not args.skip_correctness:
        correctness(min(ns[0], 2048), args.m, WINDOW, dev)
        probe_dtype_precision(dev, m=min(args.m, 512), k=WINDOW)

    print("\n=== device time per call (bf16, forward and forward+weight-grad) ===")
    head = (
        f"{'N':>8} {'conv fwd':>10} {'lin fwd':>10} {'fwd gain':>9} {'conv f+b':>10} {'lin f+b':>10} {'step gain':>10}"
    )
    print(head)
    results = [bench(n, args.m, WINDOW, dev, args.iters, args.warmup) for n in ns]
    for r in results:
        print(
            f"{r['n']:>8} {r['conv_fwd']:>10.3f} {r['lin_fwd']:>10.3f} "
            f"{r['conv_fwd'] / r['lin_fwd']:>8.1f}x {r['conv_fwd_bwd']:>10.3f} {r['lin_fwd_bwd']:>10.3f} "
            f"{r['conv_fwd_bwd'] / r['lin_fwd_bwd']:>9.1f}x"
        )
    print("(ms/call; fwd = one GEMM-equivalent of 2*N*M*K FLOPs, bwd = weight-grad GEMM only)")

    print("\n=== effective TFLOP/s ===")
    print(f"{'N':>8} {'conv fwd':>10} {'lin fwd':>10} {'conv wgrad':>11} {'lin wgrad':>10}")
    for r in results:
        f = r["flops"] / 1e9
        print(
            f"{r['n']:>8} {f / r['conv_fwd']:>10.1f} {f / r['lin_fwd']:>10.1f} "
            f"{f / r['conv_bwd']:>11.1f} {f / r['lin_bwd']:>10.1f}"
        )

    if args.profile:
        for kind in ("conv3d", "linear"):
            profile_kernels(kind, ns[-1], args.m, WINDOW, dev)

    return 0


if __name__ == "__main__":
    sys.exit(main())
