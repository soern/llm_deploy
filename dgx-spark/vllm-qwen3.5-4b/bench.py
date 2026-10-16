"""Streaming TTFT / decode TPS benchmark + MTP acceptance against an OpenAI-shaped vLLM server.

    python3 bench.py [base_url] [model] [sizes] [kind]

kind = code (default): a real coding task after the filler, so the MTP
       acceptance rate is close to what agent work sees.
       story: the Gemma deploy's filler-word story (comparable with its numbers;
       random text, so MTP acceptance is pessimistic).
"""
import json
import random
import re
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://sparky.lan:8006"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.5-4b"
SIZES = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else "1000,16000,100000").split(",")]
KIND = sys.argv[4] if len(sys.argv) > 4 else "code"
RUNS = 2
MAX_TOKENS = 512

WORDS = ("river mountain engine signal garden copper winter lantern harbor "
         "orbit meadow canvas thunder pepper silver forest window bridge "
         "marble violet rocket shadow candle ocean pillow desert falcon").split()

TASKS = {
    "code": "Ignore the text above. Write a complete Python module that implements an LRU cache "
            "with TTL expiry, thread safety and a decorator interface, with type hints, docstrings "
            "and pytest unit tests.",
    "story": "Write a long story using as many of the words above as possible.",
}


def filler(n_words, rng):
    out = []
    for i in range(n_words):
        out.append(rng.choice(WORDS))
        if i % 12 == 11:
            out.append(f"{rng.randint(0, 9999)}.")
    return " ".join(out)


def spec_counters():
    """(drafted, accepted) token totals from vLLM's Prometheus metrics; (0, 0) without MTP
    or without vLLM metrics (llama-server reports per-request "timings" instead)."""
    try:
        text = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    except OSError:
        return 0.0, 0.0
    totals = {}
    for name in ("num_draft_tokens", "num_accepted_tokens"):
        pat = re.compile(rf"^vllm:spec_decode_{name}_total(?:\{{[^}}]*\}})? ([0-9.e+]+)$", re.M)
        totals[name] = sum(float(v) for v in pat.findall(text))
    return totals["num_draft_tokens"], totals["num_accepted_tokens"]


def run(target_tokens, seed):
    rng = random.Random(seed)
    # ~1.15 tokens per filler word incl. numbers; corrected by the usage report.
    text = f"[run {seed} {time.time()}] " + filler(int(target_tokens / 1.15), rng)
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": text + "\n\n" + TASKS[KIND]}],
        "max_tokens": MAX_TOKENS,
        "min_tokens": MAX_TOKENS,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d0, a0 = spec_counters()
    t0 = time.perf_counter()
    t_first = None
    usage = None
    timings = None
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            if ev.get("timings"):
                timings = ev["timings"]
            for ch in ev.get("choices", []):
                delta = ch.get("delta", {})
                # Thinking tokens arrive as reasoning_content (--reasoning-parser).
                if t_first is None and (delta.get("content") or delta.get("reasoning_content")
                                        or delta.get("reasoning")):
                    t_first = time.perf_counter()
    t_end = time.perf_counter()
    d1, a1 = spec_counters()
    ptok, ctok = usage["prompt_tokens"], usage["completion_tokens"]
    ttft = t_first - t0
    tps = (ctok - 1) / (t_end - t_first)
    accept = (a1 - a0) / (d1 - d0) if d1 > d0 else None
    if accept is None and timings and timings.get("draft_n"):
        accept = timings["draft_n_accepted"] / timings["draft_n"]
    return ptok, ctok, ttft, tps, ptok / ttft, accept


def main():
    print(f"warmup ... (kind={KIND})", flush=True)
    run(200, 0)
    print(f"{'prompt tok':>10} {'out tok':>7} {'TTFT s':>7} {'prefill tok/s':>13} {'decode tok/s':>12} {'MTP accept':>10}")
    for size in SIZES:
        for r in range(RUNS):
            ptok, ctok, ttft, tps, pf, acc = run(size, size * 10 + r + 1)
            acc_s = f"{acc:.0%}" if acc is not None else "-"
            print(f"{ptok:>10} {ctok:>7} {ttft:>7.2f} {pf:>13.0f} {tps:>12.1f} {acc_s:>10}", flush=True)


if __name__ == "__main__":
    main()
