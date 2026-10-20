"""Measure TTFT and decode speed at several prompt lengths against a vLLM server.

Usage: python3 bench.py http://127.0.0.1:8000 16384 32768 65536 131072 196608
"""
import glob
import json
import os
import sys
import time
import urllib.request
import uuid

BASE = sys.argv[1].rstrip("/")
SIZES = [int(s) for s in sys.argv[2:]]
OUT_TOKENS = 256
QUESTION = "\n\nExplain in plain prose what the code above does, module by module."


def post(path, body, timeout=60):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(req, timeout=timeout)


model = json.load(urllib.request.urlopen(BASE + "/v1/models"))["data"][0]["id"]


def count_tokens(text):
    with post("/tokenize", {"model": model, "prompt": text}) as r:
        return json.load(r)["count"]


# Filler: Python stdlib source, i.e. varied real text rather than repetition.
corpus = []
for path in sorted(glob.glob(os.path.join(os.path.dirname(os.__file__), "*.py"))):
    try:
        corpus.append(open(path, encoding="utf-8").read())
    except (UnicodeDecodeError, OSError):
        pass
corpus = "\n".join(corpus)

# Characters per token, calibrated on a sample.
sample = corpus[:200_000]
chars_per_tok = len(sample) / count_tokens(sample)


def make_prompt(n_tokens):
    """Unique prefix (defeats prefix caching) + filler trimmed to n_tokens total."""
    prefix = f"[run {uuid.uuid4()}]\n"
    budget = n_tokens - count_tokens(prefix + QUESTION) - 20  # chat template overhead
    lo, hi = 0, min(len(corpus), int(budget * chars_per_tok * 1.3))
    # binary search on characters so the filler hits the token budget
    while hi - lo > 200:
        mid = (lo + hi) // 2
        if count_tokens(corpus[:mid]) <= budget:
            lo = mid
        else:
            hi = mid
    return prefix + corpus[:lo] + QUESTION


def run(prompt):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": OUT_TOKENS,
        "ignore_eos": True,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    t_first = None
    usage = None
    with post("/v1/chat/completions", body, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for c in chunk.get("choices", []):
                d = c.get("delta", {})
                if t_first is None and (d.get("content") or d.get("reasoning_content") or d.get("reasoning")):
                    t_first = time.perf_counter()
    t_end = time.perf_counter()
    out = usage["completion_tokens"]
    return {
        "prompt_tokens": usage["prompt_tokens"],
        "ttft_s": t_first - t0,
        "prefill_tok_s": usage["prompt_tokens"] / (t_first - t0),
        "decode_tok_s": (out - 1) / (t_end - t_first),
        "out_tokens": out,
    }


print(f"model={model} chars/token={chars_per_tok:.2f}", flush=True)
print("warmup:", run("Say hi." ), flush=True)
results = []
for n in SIZES:
    res = run(make_prompt(n))
    res["target"] = n
    results.append(res)
    print(json.dumps(res), flush=True)

print("\n| Context | Prompt tokens | TTFT | Prefill | Decode |")
print("|---|---|---|---|---|")
for r in results:
    print(
        f"| {r['target'] // 1024}K | {r['prompt_tokens']:,} | {r['ttft_s']:.1f} s | "
        f"{r['prefill_tok_s']:,.0f} tok/s | {r['decode_tok_s']:.1f} tok/s |"
    )
