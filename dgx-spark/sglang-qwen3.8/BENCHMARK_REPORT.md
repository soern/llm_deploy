# Qwen3.8-27B on a single DGX Spark — Benchmark Report & Delta

**Source:** NVIDIA Developer Forums thread #380957 ("Qwen3.8-27B benchmarking on
one DGX Spark: DFlash2 beat vLLM MTP, and greedy beat the thinking sampler").
Figures are third-party-reported; SGLang's official cookbook only confirms the
DGX Spark cells "boot and serve," it does not publish throughput.

## 1. Headline result

**Winning config: SGLang + DFlash2 (DSpark) + greedy decode**, Qwen3.8-27B NVFP4,
one DGX Spark (GB10, sm_121a, 128GB unified). It beats vLLM's MTP, which beats
the thinking sampler.

## 2. Reported decode TPS (the winning config)

**Ama5u (thread author):**
| Workload   | DFlash2 | vLLM+MTP | Speedup |
|------------|---------|----------|---------|
| Code       | 32.6    | 19.8     | ~1.6×   |
| Reasoning  | 47.3    | 25.4     | ~1.9×   |
| Prose      | ~21     | ~13      | ~1.6×   |
| Math       | 55.8    | —        | —       |

End-to-end on the 69-scenario tool-calling suite: **929s vs 2386s** (~3.4×;
also wins on latency — median turn 3.6s vs 7.8s).

**xkm121 (independent re-run):**
| Workload  | DFlash2 | MTP  | Ratio |
|-----------|---------|------|-------|
| Q&A       | 47.3    | 29.3 | 1.61× |
| Code      | 43.6    | 28.5 | 1.53× |
| JSON      | 57.2    | 34.5 | 1.66× |
| All tests | 46.9    | 29.8 | 1.57× |

**Realistic expectation: ~mid-40s tok/s sustained** for general use on one
Spark. <30 is a red flag; ~55+ appears on easy/short outputs.

## 3. The winning config (reference)

- `--speculative-algorithm DSPARK` (i.e. DFlash2) with the matching DSpark draft model
- `--mem-fraction-static 0.50`
- 262K context
- SGLang at commit `1cf2b8c` / PR #35496+
- Greedy decode (NOT the thinking sampler)
- `--sleep-on-idle` (forum only; keeps an idle server from pegging a CPU core
  at 100% — relevant for a persistent service, not a benchmark)

## 4. Delta vs the current `start.sh` in this folder

**Good news: this deploy is already on the winning engine.** `start.sh` already
sets `--speculative-algorithm DSPARK` + the DSpark draft model, so the 1.5–1.7×
DFlash2 win over MTP is already in place. Below is the per-flag status.

| Flag                          | Current `start.sh` | Winning reference | Action |
|-------------------------------|--------------------|-------------------|--------|
| `--speculative-algorithm`     | `DSPARK`           | `DSPARK` (DFlash2)| ✅ No change |
| `--speculative-draft-model-path` | DSpark draft   | DSpark draft      | ✅ No change |
| `--speculative-dspark-block-size` | `7`              | not stated in thread | ✅ keep (block size 7 is DSpark's default) |
| `--mem-fraction-static`       | `0.80`             | `0.50`            | ⚠️ **differs** — see below |
| Context / max context         | (not pinned here)  | 262K             | ⚠️ confirm context target matches |
| Decode sampler                | greedy (default)   | greedy            | ✅ No change |
| `--sleep-on-idle`             | not set            | set (persistent svc) | ⚠️ **optional** — see below |

### Delta detail

**⚠️ `MEM_FRACTION_STATIC`: 0.80 (now) vs 0.50 (forum).**
This is the one real disagreement. The forum reference ran at **0.50**; this
`start.sh` runs at **0.80** because it follows SGLang's *official cookbook*
(SM120/SM121 section), which validated 0.80 as the safe ceiling across all 48
config combos on this exact hardware and documented that 0.85 gets SIGTERMed
by earlyoom. The 0.50 in the forum post was a *third-party* recipe value, and
the README here explicitly documents that 0.80 supersedes it.
→ **Recommendation: KEEP 0.80.** The tok/s figures are decode-throughput
numbers and do not depend on mem-fraction; the higher fraction only buys more
KV/mamba headroom. Do not drop to 0.50 just to match the thread — it would
reduce available context/running-requests with no throughput upside.

**⚠️ `--sleep-on-idle`: not set (now).**
Not a benchmark knob — it's a *service* knob. Adding it means an idle server
releases its CPU instead of spinning a core at 100%. The current `start.sh`
does not set it.
→ **Action: add it only if this Spark runs as a persistent always-on service.**
Flag it on in `SGLANG_ARGS` as `--sleep-on-idle`. No effect on measured tok/s.

**⚠️ Context target (262K).**
Confirm your serving context matches the 262K the benchmark used. If you're
serving a smaller context, the effective headroom at mem-fraction 0.80 is even
better than the forum's 0.50/262K combo. No change needed if you're already at
or below 262K.

### Net Delta
- **Engine/algorithm: no change** (already DFlash2/DSpark).
- **`MEM_FRACTION_STATIC`: keep 0.80** (do not copy the forum's 0.50).
- **`--sleep-on-idle`: optional** — add only for persistent-service use.
- **Context: verify ≤ 262K.**

## 5. Caveats
1. Numbers are for the specific NVFP4 quant + 262K context in the thread.
   DFlash2's lead shifts with a different quant or longer context.
2. The published `lmsysorg/sglang:qwen38-27b` image predates the quantized
   `lm_head` selector and will not boot NVFP4 — this deploy's
   `dev-qwen38-27b-dflash2` image is required.
3. `--sleep-on-idle` was not re-verified against the SGLang source (skipped per
   request). Confirm it parses on the exact image commit before relying on it.
