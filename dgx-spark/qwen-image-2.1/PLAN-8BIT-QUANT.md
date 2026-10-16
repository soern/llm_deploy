# Plan: Switch Qwen-Image-2.1 to 8-bit quantization

**Date:** 2026-09-22 · **Status:** Plan, pending user confirmation (model path + INT8 vs FP8 choice)
**Update 2026-10-04:** Only the FP8 text encoder was implemented (now the default, `TEXT_ENCODER_QUANT=fp8`, downloaded via the HF hub instead of a host mount). The DiT stays bf16; the INT8 DiT part of this plan is not being pursued.
**Motivation:** Second OOM-kill of the image container; free RAM on the shared 128 GB pool dropped to ~6.4 GB before the kill. bf16 pipeline sits at ~31 GB resident + ~6 GB transient activation spike at 2048². Quantizing the DiT (and optionally the text encoder) cuts the resident footprint by ~7–15 GB, giving real headroom again.

## Source material (verified)

`unsloth/Qwen-Image-2.1-FP8` (diffusers-format, safetensors only, no pickle code on load):

| File | Size | Replaces (bf16/fp32) | Savings |
|---|---|---|---|
| `Qwen-Image-2.1-INT8.safetensors` | 7.26 GB | `transformer` 14.23 GB | ~7 GB |
| `Qwen-Image-2.1-FP8.safetensors` | 7.12 GB | `transformer` 14.23 GB | ~7 GB |
| `Qwen-Image-2.1-text_encoder-FP8.safetensors` | 9.39 GB | `text_encoder` 17.5 GB | ~8 GB |
| `vae/qwen_image_2.1_vae_bf16.safetensors` | 0.63 GB | `vae` (fp32) | ~0.3 GB |

Same-seed LPIPS vs bf16: **INT8 0.064** (near-lossless, the shipped scheme), **FP8 0.112**.
→ **Recommendation: INT8 for the DiT** (best quality at same memory class). Text-encoder quant is FP8-only in that repo → treat as optional phase 2.

Expected effect on resident memory: **~31 GB → ~24 GB (INT8 DiT only)** or **~16 GB (INT8 DiT + FP8 TE)**.

Requires diffusers main (already pinned in the Dockerfile) — quantized single-file loading postdates 0.40.

## Open questions (need answers before implementing)

1. **Exact path on the backend host** where the quantized file(s) sit (e.g. `/mnt/models/Qwen-Image-2.1-INT8.safetensors` or a local repo dir).
2. **INT8 or FP8?** Plan assumes INT8 (shipped scheme, best quality); FP8 is a one-line file swap if preferred.
3. Do we want the FP8 text encoder in scope now, or keep TE in bf16 for the first cut?

## Implementation plan

### Phase 1 — Mount the model into the container
1. `docker-compose.yml`: add a volume mount for the host-side model file/dir → read-only, e.g.
   `- /host/path:/models/qwen-image-2.1:ro` (mirror the pattern the flux2-klein stack uses for baked/local weights — check `deploy/dgx-spark/flux2-klein/docker-compose.yml` for the convention).
   No model download at runtime, no image rebuild needed for the weights.

### Phase 2 — server.py loading logic
2. Make `QUANT_MODE` an env knob (currently hardcoded `"none"`): `none` (default, current behavior) | `int8` | `fp8`.
3. New env vars: `TRANSFORMER_PATH` (e.g. `/models/qwen-image-2.1/Qwen-Image-2.1-INT8.safetensors`), optionally `TEXT_ENCODER_PATH`, `VAE_PATH`.
4. In `_load_pipeline()`:
   - `QUANT_MODE=none` → unchanged: `QwenImage21Pipeline.from_pretrained(IMAGE_MODEL_ID, torch_dtype=torch.bfloat16, token=HF_TOKEN).to("cuda")`.
   - Else → load base pipeline as today, then swap components:
     - `pipe.transformer = <QwenImage transformer class>.from_single_file(TRANSFORMER_PATH)` (diffusers detects INT8/FP8 from the safetensors via its quantization manager; verify at test time, fall back to an explicit `quantization_config` if needed).
     - Phase 2 only: `pipe.text_encoder` and `pipe.vae` swapped the same way.
   - Keep `.to("cuda")` / device handling as-is (unified memory, same physical pool).
   - Update the load log line to print the actual model source.
5. `/health` + `/v1/models`: report real `quant_mode` (already wired — just no longer hardcoded).
6. **No client changes** in `src/nixclaw_agent/imagegen.py` — OpenAI-shaped HTTP API unchanged.

### Phase 3 — Deploy
7. User redeploys updated `docker-compose.yml` + `server.py` on `gx10-811e.lan` (start.sh already re-deployed tonight).
8. Confirm startup logs: model source, no dtype/quant warnings, `torch.cuda` memory report.

### Phase 4 — Verify (bench script, conservative pacing per tonight's thermal OOMs)
9. `/health`: `quant_mode=int8`, `resident_memory_mb` ≈ 20–25 GB (down from ~31 GB).
10. **Quality A/B** (the gate): run `bench_backend.py --quality ... --quality-steps 40` at 2048², seed 42 — byte-identical prompt/seed → visual diff against `tmp/20260922-024154-quality-2048x2048-40steps.png` (bf16 reference). Also run 2–3 of the fancy-batch prompts.
11. Speed sanity: one 40-step 2048² timing vs the 276 s baseline (INT8 may be marginally faster or neutral — no big speedup expected, and speed work is parked anyway).
12. Stress/soak: 2–3 consecutive 40-step generations with cooldown; watch host free RAM on the box to confirm the OOM headroom actually improved.

### Rollback
13. `QUANT_MODE=none` + container restart → back to bf16, zero rebuild.

## Risks
- **Single-file quant load API**: diffusers main must detect INT8/FP8 from the safetensors; if it doesn't, we pass an explicit quantization config (small code change, still no client impact). Caught at step 9–10.
- **Quality regression**: INT8 LPIPS 0.064 is near-lossless on unsloth's samples, but our gate is the same-seed visual diff at step 10, not their samples.
- **OOM is only mitigated, not cured**: ~7–15 GB extra headroom reduces recurrence; the admission guard still reads `pool_free_mb` which includes reclaimable page cache (see host-tuning follow-up: dirty-ratio sysctls / `MIN_FREE_VRAM_MB` should grow to ~18–20 GB to cover the +6 GB transient spike). Quantization and that guard fix are independent — recommend doing the guard fix regardless.
- **Thermal**: keep the `-lgc 300,2100` clock lock and spaced-out test runs; soak test at step 12 runs with cooldowns.

## Not in scope
- FP8 DiT as primary (INT8 preferred; FP8 kept as drop-in alternative)
- FP8 text encoder (phase 2, optional, after INT8 proves itself)
- Any client-side changes (`imagegen.py`, `config/agent.yaml`)
- Steps reduction / FP8 speed work (parked per user decision)
