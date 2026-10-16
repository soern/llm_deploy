#!/usr/bin/env python3
"""Diagnostic probe for Qwen-Image-2.1 slowness on the DGX Spark (GB10).

Answers, with measurements rather than guesses:
  1. Is the box actually compute-bound, or are we on a fallback attention
     kernel?  (10.5 s/step at 2048^2 implies ~32 TFLOPS effective, roughly
     10% of what GB10 should manage in bf16 - so this is the top suspect.)
  2. Which SDPA backends even work on sm_121a, and how fast is each at the
     real attention shape?
  3. Does torch.compile work here, and what does it buy?
  4. Is FP8 compute faster than BF16 on this box (i.e. is quantizing the DiT
     a *speed* lever, or only a memory lever as the vLLM cookbook claims)?
  5. How does end-to-end latency split into fixed overhead (text encode +
     VAE decode) vs per-step cost, and how does it scale with resolution?

Two independent modes - run either or both:

  # A) microbenchmarks: run INSIDE the model container (needs torch).
  #    Loads no model weights, uses ~2 GB VRAM, takes ~2 minutes.
  docker cp bench_backend.py qwen-image-2.1:/tmp/bench_backend.py
  docker exec qwen-image-2.1 python3 /tmp/bench_backend.py --micro

  # B) end-to-end timing against the running server over HTTP.
  #    Sends a handful of SHORT (low-step) requests; ~5-10 min total.
  python3 bench_backend.py --http http://localhost:8002

Paste the whole output back. Nothing here writes files or changes the
running service.
"""

import argparse
import json
import os
import sys
import time
from typing import Optional

# Real geometry from Qwen/Qwen-Image-2.1 transformer/config.json:
NUM_LAYERS = 32
NUM_HEADS = 32
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM  # 4096
MLP_RATIO = 3
# One token per 16x16 pixel tile (vae_scale_factor=16, patch_size=1).
TOKENS = {"1024x1024": (1024 // 16) ** 2, "1536x1536": (1536 // 16) ** 2, "2048x2048": (2048 // 16) ** 2}


def hr(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


# --------------------------------------------------------------------------
# A) Microbenchmarks
# --------------------------------------------------------------------------
def bench_micro() -> None:
    import torch

    hr("1. ENVIRONMENT")
    print(f"torch              : {torch.__version__}")
    print(f"cuda (torch build) : {torch.version.cuda}")
    try:
        import diffusers

        print(f"diffusers          : {diffusers.__version__}")
    except Exception as exc:
        print(f"diffusers          : MISSING ({exc})")
    try:
        import transformers

        print(f"transformers       : {transformers.__version__}")
    except Exception as exc:
        print(f"transformers       : MISSING ({exc})")
    try:
        import flash_attn

        print(f"flash_attn         : {flash_attn.__version__}  <- installed")
    except Exception:
        print("flash_attn         : NOT installed (torch SDPA may fall back)")

    if not torch.cuda.is_available():
        print("\nCUDA NOT AVAILABLE - aborting microbenchmarks")
        return
    props = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"device             : {props.name}")
    print(f"capability         : sm_{cap[0]}{cap[1]}")
    print(f"SMs                : {props.multi_processor_count}")
    print(f"total memory       : {props.total_memory / 1024**3:.1f} GiB")
    print(f"memory in use now  : {torch.cuda.memory_reserved() / 1024**3:.1f} GiB reserved")

    def timed(fn, warmup=2, iters=5):
        """Return best-of milliseconds (best-of rejects co-tenant interference)."""
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        best = float("inf")
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            best = min(best, time.perf_counter() - t0)
        return best * 1000.0

    # ---------------------------------------------------------------- GEMM
    hr("2. RAW GEMM CEILING (what the hardware can do)")
    print("Establishes the bf16 compute ceiling. Compare everything below to this.\n")
    gemm_tflops = {}
    for n in (4096, 8192):
        try:
            a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
            ms = timed(lambda: torch.mm(a, b))
            tflops = (2 * n**3) / (ms / 1000) / 1e12
            gemm_tflops[n] = tflops
            print(f"  bf16 {n}x{n} matmul : {ms:8.2f} ms  -> {tflops:7.1f} TFLOPS")
            del a, b
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"  bf16 {n}: FAILED: {exc}")
    peak = max(gemm_tflops.values()) if gemm_tflops else 0.0

    # ----------------------------------------------------------------- FP8
    hr("3. FP8 vs BF16 COMPUTE (is quantization a SPEED lever here?)")
    print("The vLLM cookbook says FP8 buys memory, not speed - but that was on")
    print("bandwidth-bound GB200. If FP8 is much faster here, option #6 is live.\n")
    try:
        n = 8192
        af = torch.randn(n, n, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        bf = torch.randn(n, n, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn).t().contiguous().t()
        sa = torch.tensor(1.0, device="cuda")
        sb = torch.tensor(1.0, device="cuda")
        ms = timed(lambda: torch._scaled_mm(af, bf, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16))
        tf = (2 * n**3) / (ms / 1000) / 1e12
        ratio = tf / gemm_tflops.get(8192, tf)
        print(f"  fp8_e4m3 {n}x{n}   : {ms:8.2f} ms  -> {tf:7.1f} TFLOPS  ({ratio:.2f}x vs bf16)")
        del af, bf
        torch.cuda.empty_cache()
    except Exception as exc:
        print(f"  fp8 _scaled_mm FAILED: {type(exc).__name__}: {exc}")

    # ----------------------------------------------------------- ATTENTION
    hr("4. ATTENTION BACKENDS AT THE REAL SHAPE  <-- THE KEY TEST")
    print(f"Shape: batch=1, heads={NUM_HEADS}, head_dim={HEAD_DIM}, bf16")
    print("seq 4096 = 1024x1024 output, seq 16384 = 2048x2048 output.")
    print("'default' is what diffusers actually gets today.\n")

    from torch.nn.attention import SDPBackend, sdpa_kernel

    backends = [
        ("default", None),
        ("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
        ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
        ("CUDNN_ATTENTION", SDPBackend.CUDNN_ATTENTION),
        ("MATH", SDPBackend.MATH),
    ]

    for label, seq in (("1024x1024", 4096), ("2048x2048", 16384)):
        print(f"--- seq={seq} ({label}) ---")
        try:
            q = torch.randn(1, NUM_HEADS, seq, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(1, NUM_HEADS, seq, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(1, NUM_HEADS, seq, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        except Exception as exc:
            print(f"  could not allocate: {exc}")
            continue
        # FLOPs for one attention: 2 matmuls, 2 flops each element.
        attn_flops = 4 * seq * seq * HEAD_DIM * NUM_HEADS
        for name, backend in backends:
            # MATH materializes seq^2 per head: at seq=16384 that is ~17 GiB. Skip.
            if backend == SDPBackend.MATH and seq > 8192:
                print(f"  {name:22}: skipped (would allocate ~17 GiB attn matrix)")
                continue
            try:
                import torch.nn.functional as F

                if backend is None:
                    fn = lambda: F.scaled_dot_product_attention(q, k, v)
                else:
                    def fn(b=backend):
                        with sdpa_kernel(b):
                            return F.scaled_dot_product_attention(q, k, v)
                ms = timed(fn, warmup=1, iters=3)
                tf = attn_flops / (ms / 1000) / 1e12
                pct = f"{100 * tf / peak:4.0f}% of GEMM peak" if peak else ""
                print(f"  {name:22}: {ms:8.2f} ms  -> {tf:7.1f} TFLOPS  {pct}")
            except Exception as exc:
                msg = str(exc).split("\n")[0][:90]
                print(f"  {name:22}: UNAVAILABLE ({type(exc).__name__}: {msg})")
        del q, k, v
        torch.cuda.empty_cache()
        print()

    # ------------------------------------------------- MASKED ATTENTION
    hr("4b. MASKED ATTENTION  <-- PRIME SUSPECT FOR THE UNEXPLAINED 2.5x")
    print("QwenImage21 passes img_mask / encoder_hidden_states_mask for its")
    print("mixed-granularity attention. A non-None attn_mask DISQUALIFIES the")
    print("flash backend, forcing the ~2.5x slower efficient kernel. If the")
    print("masked rows below are much slower than unmasked, that is the gap.\n")

    import torch.nn.functional as F

    for label, seq in (("1024x1024", 4096), ("2048x2048", 16384)):
        print(f"--- seq={seq} ({label}) ---")
        try:
            q = torch.randn(1, NUM_HEADS, seq, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
            k = torch.randn(1, NUM_HEADS, seq, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(1, NUM_HEADS, seq, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        except Exception as exc:
            print(f"  could not allocate: {exc}")
            continue
        attn_flops = 4 * seq * seq * HEAD_DIM * NUM_HEADS

        # Full [1,1,S,S] bool mask: what a per-token mixed-granularity mask costs.
        # Broadcast [1,1,1,S] mask: what a cheap key-padding mask costs.
        variants = [("no mask (baseline)", None)]
        try:
            variants.append(("full SxS bool mask", torch.ones(1, 1, seq, seq, dtype=torch.bool, device="cuda")))
        except Exception as exc:
            print(f"  full mask alloc failed: {exc}")
        variants.append(("broadcast 1xS mask", torch.ones(1, 1, 1, seq, dtype=torch.bool, device="cuda")))

        for vname, mask in variants:
            for bname, backend in (("default", None), ("cudnn", SDPBackend.CUDNN_ATTENTION)):
                try:
                    if backend is None:
                        fn = lambda m=mask: F.scaled_dot_product_attention(q, k, v, attn_mask=m)
                    else:
                        def fn(m=mask, b=backend):
                            with sdpa_kernel(b):
                                return F.scaled_dot_product_attention(q, k, v, attn_mask=m)
                    ms = timed(fn, warmup=1, iters=3)
                    tf = attn_flops / (ms / 1000) / 1e12
                    print(f"  {vname:20} [{bname:7}]: {ms:8.2f} ms  -> {tf:7.1f} TFLOPS")
                except Exception as exc:
                    msg = str(exc).split("\n")[0][:70]
                    print(f"  {vname:20} [{bname:7}]: UNAVAILABLE ({type(exc).__name__}: {msg})")
            del mask
        del q, k, v
        torch.cuda.empty_cache()
        print()

    # ------------------------------------------------- PER-STEP PROJECTION
    hr("5. PROJECTED PER-STEP COST FROM FLOP COUNTS")
    print("Compares the theoretical DiT step cost against your observed 10.5 s/step.\n")
    per_layer_params = 4 * HIDDEN**2 + 2 * MLP_RATIO * HIDDEN**2
    for label, seq in TOKENS.items():
        linear = 2 * per_layer_params * NUM_LAYERS * seq
        attn = 4 * seq * seq * HEAD_DIM * NUM_HEADS * NUM_LAYERS
        total = linear + attn
        print(
            f"  {label}: seq={seq:>5}  linear={linear/1e12:6.1f} + attn={attn/1e12:6.1f} "
            f"= {total/1e12:6.1f} TFLOP/step"
        )
        if peak:
            print(f"      at measured GEMM peak ({peak:.0f} TFLOPS): {total/1e12/peak:6.2f} s/step floor")

    # ------------------------------------------------------ TORCH.COMPILE
    hr("6. TORCH.COMPILE SMOKE TEST (does inductor work on sm_121a?)")
    print("One representative transformer block, eager vs compiled.\n")
    try:
        import torch.nn as nn

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.n1 = nn.LayerNorm(HIDDEN)
                self.qkv = nn.Linear(HIDDEN, 3 * HIDDEN)
                self.proj = nn.Linear(HIDDEN, HIDDEN)
                self.n2 = nn.LayerNorm(HIDDEN)
                self.fc1 = nn.Linear(HIDDEN, MLP_RATIO * HIDDEN)
                self.fc2 = nn.Linear(MLP_RATIO * HIDDEN, HIDDEN)

            def forward(self, x):
                import torch.nn.functional as F

                b, s, _ = x.shape
                h = self.n1(x)
                q, k, v = self.qkv(h).view(b, s, 3, NUM_HEADS, HEAD_DIM).permute(2, 0, 3, 1, 4)
                a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, s, HIDDEN)
                x = x + self.proj(a)
                return x + self.fc2(F.gelu(self.fc1(self.n2(x))))

        seq = 4096  # keep the smoke test cheap
        blk = Block().cuda().to(torch.bfloat16).eval()
        x = torch.randn(1, seq, HIDDEN, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            eager = timed(lambda: blk(x), warmup=2, iters=3)
            print(f"  eager    : {eager:8.2f} ms/block  -> x{NUM_LAYERS} = {eager*NUM_LAYERS/1000:5.2f} s/step")
            t0 = time.perf_counter()
            cblk = torch.compile(blk)
            cblk(x)
            torch.cuda.synchronize()
            print(f"  compile warmup: {time.perf_counter()-t0:.1f} s (one block; x32 blocks share it)")
            comp = timed(lambda: cblk(x), warmup=2, iters=3)
            print(f"  compiled : {comp:8.2f} ms/block  -> x{NUM_LAYERS} = {comp*NUM_LAYERS/1000:5.2f} s/step")
            print(f"  speedup  : {eager/comp:.2f}x")
    except Exception as exc:
        msg = str(exc).split("\n")[0][:200]
        print(f"  torch.compile FAILED: {type(exc).__name__}: {msg}")
        print("  -> option #4 (compile_repeated_blocks) is off the table on this box.")

    hr("MICROBENCHMARKS DONE")


# --------------------------------------------------------------------------
# B) End-to-end HTTP timing
# --------------------------------------------------------------------------
def _save_image(b64_png: str, out_dir: str, run_ts: str, label: str) -> Optional[str]:
    """Write one result image to <out_dir>/<run_ts>-<label>.png for eyeballing.

    Speed numbers alone cannot tell you whether a step count is acceptable, so
    every benchmarked render is kept on disk under the run's start timestamp.
    """
    import base64

    try:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{run_ts}-{label}.png")
        with open(path, "wb") as fh:
            fh.write(base64.b64decode(b64_png))
        return path
    except Exception as exc:
        print(f"    (could not save image: {type(exc).__name__}: {str(exc)[:80]})")
        return None


def bench_http(base_url: str, prompt: str, full: bool, out_dir: str, run_ts: str) -> None:
    import urllib.request

    base = base_url.rstrip("/")
    saved: list[str] = []

    def call(size: str, steps: int, tag: str = ""):
        body = json.dumps(
            {"prompt": prompt, "size": size, "n": 1, "response_format": "b64_json", "steps": steps, "seed": 42}
        ).encode()
        req = urllib.request.Request(
            f"{base}/v1/images/generations", data=body, headers={"Content-Type": "application/json"}
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=1800) as resp:
                payload = json.load(resp)
        except Exception as exc:
            return None, f"{type(exc).__name__}: {str(exc)[:120]}"
        dt = time.perf_counter() - t0
        if "error" in payload:
            return None, str(payload["error"])[:120]
        b64 = payload.get("data", [{}])[0].get("b64_json", "")
        label = f"{size}-{steps}steps" + (f"-{tag}" if tag else "")
        path = _save_image(b64, out_dir, run_ts, label) if b64 else None
        if path:
            saved.append(path)
        return dt, f"{len(b64)/1024:.0f} KB b64" + (f" -> {os.path.basename(path)}" if path else "")

    hr("7. SERVER HEALTH")
    try:
        with urllib.request.urlopen(f"{base}/health", timeout=30) as resp:
            print(json.dumps(json.load(resp), indent=2))
    except Exception as exc:
        print(f"could not reach {base}/health: {exc}")
        return

    hr("8. STEP SCALING (separates fixed overhead from per-step cost)")
    print("Fits latency = fixed + steps * per_step. 'fixed' is text encode +")
    print("VAE decode + transport; 'per_step' is the DiT denoising loop.\n")
    results = {}
    size = "2048x2048"
    for steps in (4, 12):
        dt, info = call(size, steps, tag="stepscale")
        results[steps] = dt
        print(f"  {size} @ {steps:>3} steps : " + (f"{dt:7.1f} s   ({info})" if dt else f"FAILED: {info}"))
    if results.get(4) and results.get(12):
        per_step = (results[12] - results[4]) / 8
        fixed = results[4] - 4 * per_step
        print(f"\n  -> per-step : {per_step:6.2f} s")
        print(f"  -> fixed    : {fixed:6.2f} s  (text encode + VAE decode + transport)")
        print(f"  -> projected 40 steps: {fixed + 40*per_step:6.1f} s  ({(fixed+40*per_step)/60:.1f} min)")
        if fixed > 60:
            print("  -> NOTE: large fixed cost; the text encoder/VAE is a target too, not just steps.")

    hr("9. RESOLUTION SCALING (at fixed 4 steps)")
    print("Attention is quadratic in tokens, so this shows what dropping")
    print("resolution actually buys. Token counts: " + ", ".join(f"{k}={v}" for k, v in TOKENS.items()) + "\n")
    base_dt = None
    for size in ("1024x1024", "1536x1536", "2048x2048"):
        dt, info = call(size, 4, tag="resscale")
        if dt:
            if size == "1024x1024":
                base_dt = dt
            rel = f"{dt/base_dt:5.2f}x vs 1024" if base_dt else ""
            print(f"  {size} @ 4 steps : {dt:7.1f} s   {rel}")
        else:
            print(f"  {size} @ 4 steps : FAILED: {info}")

    if full:
        hr("10. WARMUP CHECK (is the first call penalized?)")
        for i in (1, 2):
            dt, info = call("2048x2048", 4, tag=f"warmup{i}")
            print(f"  run {i}: " + (f"{dt:7.1f} s" if dt else f"FAILED: {info}"))

    hr("HTTP BENCHMARKS DONE")
    print("NOTE: these runs are intentionally undersampled (4-12 steps), so the")
    print("images show speed-tuning artifacts, NOT the quality of a 40-step run.")
    print("Use --quality to compare step counts at a fair, fixed seed.\n")
    if saved:
        print(f"{len(saved)} image(s) written to {os.path.abspath(out_dir)}/")
        for path in saved:
            print(f"  {os.path.basename(path)}")


def bench_quality(base_url: str, prompt: str, size: str, steps_list: list, out_dir: str, run_ts: str) -> None:
    """Render the same prompt+seed at several step counts, for visual A/B.

    Dropping NUM_INFERENCE_STEPS from the official 40 is the cheapest speed
    lever available (1.65x at 24), but it is a QUALITY tradeoff and no
    benchmark can decide it for you. Fixed seed so only the step count varies.
    """
    import urllib.request

    base = base_url.rstrip("/")
    hr("11. QUALITY SWEEP (same seed, varying steps - inspect these by eye)")
    print(f"{size}, seed 42, prompt: {prompt!r}")
    print("Official default is 40 steps. Compare the files pairwise and pick")
    print("the lowest count you cannot tell apart from 40.\n")

    for steps in steps_list:
        body = json.dumps(
            {"prompt": prompt, "size": size, "n": 1, "response_format": "b64_json", "steps": steps, "seed": 42}
        ).encode()
        req = urllib.request.Request(
            f"{base}/v1/images/generations", data=body, headers={"Content-Type": "application/json"}
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=3600) as resp:
                payload = json.load(resp)
        except Exception as exc:
            print(f"  {steps:>3} steps : FAILED: {type(exc).__name__}: {str(exc)[:100]}")
            continue
        dt = time.perf_counter() - t0
        if "error" in payload:
            print(f"  {steps:>3} steps : FAILED: {str(payload['error'])[:100]}")
            continue
        b64 = payload.get("data", [{}])[0].get("b64_json", "")
        path = _save_image(b64, out_dir, run_ts, f"quality-{size}-{steps}steps") if b64 else None
        print(f"  {steps:>3} steps : {dt:7.1f} s  -> {os.path.basename(path) if path else '(not saved)'}")

    print(f"\nWritten to {os.path.abspath(out_dir)}/")


def bench_profile(base_url: str, prompt: str, size: str, steps: int) -> None:
    """Ask the server to profile itself and print the hottest kernels.

    Per-step cost runs ~1.55x above the block-benchmark floor at BOTH 1024
    and 2048, so the overhead is a constant factor, not attention scaling.
    This attributes it to actual kernels.
    """
    import urllib.request

    base = base_url.rstrip("/")
    hr("10. KERNEL PROFILE (what the unexplained ~1.55x is actually spent on)")
    print(f"Profiling {size} @ {steps} steps on the server's loaded pipeline.")
    print("Note: profiling adds overhead, so per_step_s here is NOT a clean")
    print("latency number - only the RELATIVE kernel shares are meaningful.\n")

    body = json.dumps({"prompt": prompt, "size": size, "steps": steps}).encode()
    req = urllib.request.Request(
        f"{base}/debug/profile", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            payload = json.load(resp)
    except Exception as exc:
        print(f"profile request failed: {type(exc).__name__}: {str(exc)[:200]}")
        print("(needs a server built after the profiler change; rebuild if 404)")
        return

    if "error" in payload:
        print(f"server error: {payload['error'][:300]}")
        return

    print(f"  elapsed        : {payload.get('elapsed_s')} s ({payload.get('per_step_s')} s/step, profiled)")
    print(f"  total CUDA time: {payload.get('total_cuda_ms')} ms")
    print("\n  time by category:")
    for name, pct in (payload.get("bucket_share_pct") or {}).items():
        print(f"    {name:20}: {pct}%")
    print("\n  NOTE: op-name rows (aten::mm) and kernel-name rows (nvjet_...)")
    print("  both carry self time for the SAME work in key_averages(), so")
    print("  these percentages are relative, not an exact partition.\n")
    print("  'launch_overhead' large  -> many small per-block kernel launches;")
    print("                              CUDA graphs / larger fused ops help.")
    print("  'elementwise/other' large -> unfused norm/modulation/RoPE; fusion")
    print("                              or torch.compile on the full block.")
    print("  'gemm' dominant          -> already near hardware; only FP8")
    print("                              (2.13x measured) or fewer steps help.\n")
    print(payload.get("table", "(no table returned)"))

    hr("PROFILE DONE")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--micro", action="store_true", help="run GPU microbenchmarks (inside the container)")
    p.add_argument("--http", metavar="URL", help="run end-to-end timing against a running server")
    p.add_argument("--profile", metavar="URL", help="profile one generation on a running server and list hot kernels")
    p.add_argument("--profile-size", default="2048x2048", help="resolution for --profile (default 2048x2048)")
    p.add_argument("--profile-steps", type=int, default=2, help="steps for --profile (default 2; keep small)")
    p.add_argument("--prompt", default="A red bicycle leaning against a brick wall", help="prompt for HTTP tests")
    p.add_argument("--full", action="store_true", help="add the warmup-repeat test")
    p.add_argument("--quality", metavar="URL", help="render the same seed at several step counts for visual comparison")
    p.add_argument("--quality-size", default="2048x2048", help="resolution for --quality (default 2048x2048)")
    p.add_argument(
        "--quality-steps", default="40,32,24,16", help="comma-separated step counts for --quality (default 40,32,24,16)"
    )
    p.add_argument(
        "--out-dir", default="tmp", help="where to write result images (default: tmp/, the repo's only temp dir)"
    )
    args = p.parse_args()

    if not args.micro and not args.http and not args.profile and not args.quality:
        p.print_help()
        return 1
    run_ts = time.strftime("%Y%m%d-%H%M%S")
    print(f"qwen-image-2.1 diagnostics | {time.strftime('%Y-%m-%d %H:%M:%S')} | run id {run_ts}")
    if args.micro:
        try:
            bench_micro()
        except Exception as exc:
            import traceback

            print(f"\nMICRO BENCH CRASHED: {exc}")
            traceback.print_exc()
    if args.http:
        bench_http(args.http, args.prompt, args.full, args.out_dir, run_ts)
    if args.quality:
        try:
            steps_list = [int(s) for s in args.quality_steps.split(",") if s.strip()]
        except ValueError:
            print(f"bad --quality-steps {args.quality_steps!r}; expected e.g. 40,32,24")
            return 1
        bench_quality(args.quality, args.prompt, args.quality_size, steps_list, args.out_dir, run_ts)
    if args.profile:
        bench_profile(args.profile, args.prompt, args.profile_size, args.profile_steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
