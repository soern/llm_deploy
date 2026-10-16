# qwen-image-2.1-dgx-spark

Image-generation/editing HTTP service backed by
[Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo)
(default) or the base [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1),
running on the same DGX Spark (GB10, ARM64, sm_121a, 128 GB unified memory)
box that hosts vLLM (`deploy/dgx-spark/vllm-qwen3.6/`). Drop-in replacement for
`deploy/dgx-spark/flux2-klein/` — same OpenAI-shaped endpoints
(`/v1/images/generations`, `/v1/images/edits`, `/health`, `/v1/models`)
and same host port (default **8002**), so the nixclaw client
(`src/nixclaw_agent/imagegen.py`) works unchanged.

## Model

Qwen-Image-2.1 is a 7.1B single-stream DiT with a Qwen3-VL-8B text encoder
and a 16x RGBA autoencoder. **One model serves both text-to-image and
image-conditioned generation (editing)** — up to 10 reference images per
request — no separate edit model needed.

Component sizes (on disk, BF16):
| Component | Class | On disk |
|---|---|---|
| DiT | `QwenImage21Transformer2DModel` | 14.2 GB |
| Text encoder | `Qwen3VLForConditionalGeneration` | 17.5 GB |
| VAE | `AutoencoderKLQwenImage21` | 1.4 GB (fp32) |

The repo is **not gated** on HuggingFace; `HF_TOKEN` is accepted but optional.

### Turbo (default) vs base

`IMAGE_MODEL_ID` defaults to `Qwen/Qwen-Image-2.1-Turbo`, a distilled
fine-tune of the base model:

| | Turbo (default) | Base |
|---|---|---|
| `IMAGE_MODEL_ID` | `Qwen/Qwen-Image-2.1-Turbo` | `Qwen/Qwen-Image-2.1` |
| Steps | **8, fixed by the checkpoint** | 40 (`NUM_INFERENCE_STEPS=40`) |
| 2048² latency (est.) | ~55 s | ~163 s at 24 steps, ~4.5 min at 40 |
| Guidance | none (`TRUE_CFG_SCALE=1.0`) | none |

Same DiT architecture (different weights), same VAE architecture, and a
**byte-identical text encoder** (checked 2026-10-10 by comparing sampled
tensors across the language model, vision tower and `lm_head`; the repo
only reshards it into one file). So the FP8 encoder below applies unchanged,
and resident memory, the 2K resolutions, editing and the VAE tiling logic
are all the same.

**Fixed schedule — `steps` is ignored.** Turbo stores its 8-step sigma grid
as `sample_sigmas` in `model_index.json`. With it present the pipeline
ignores `num_inference_steps`, so `NUM_INFERENCE_STEPS`, the request's
`steps` field and the agent's `image_gen.steps` have no effect. `/health`
reports the real count (`steps: 8`, `fixed_schedule: true`).

**Needs diffusers PR #14950 (merged 2026-10-05).** An older diffusers does
not error on `sample_sigmas` — it drops the key and samples a plain
schedule at the request's step count, i.e. the distilled model on the wrong
schedule, which looks like bad image quality, not a bug. Two guards:
the Docker build asserts the support is present, and startup compares the
checkpoint's `model_index.json` with the loaded pipeline (mismatch ->
`/health` 503 with the reason). Docker caches the git-main `pip install`
layer, so an image built before 2026-10-05 fails the build check; rebuild
once with `NO_CACHE=1 ./start.sh`.

To go back to the base model:
`IMAGE_MODEL_ID=Qwen/Qwen-Image-2.1 NUM_INFERENCE_STEPS=40 ./start.sh`
(after `./stop.sh`).

### This variant: bf16 DiT + FP8 text encoder

The DiT and VAE run **unquantized (bf16)**. The Qwen3-VL text encoder is
loaded by default from
[`unsloth/Qwen-Image-2.1-FP8`](https://huggingface.co/unsloth/Qwen-Image-2.1-FP8)
(`Qwen-Image-2.1-text_encoder-FP8.safetensors`, 9.4 GB instead of 17.5 GB):
every Linear weight/bias is stored as `float8_e4m3fn` and upcast to bf16 per
layer at compute time; norms and embeddings stay bf16. Expected resident
memory drops from ~31 GB to ~23 GB (verify with `/health`
`resident_memory_mb`; `quant_mode` reports `text_encoder_fp8`). The encoder
runs once per request, so latency is essentially unchanged.

`TEXT_ENCODER_QUANT=none` restores the full-bf16 encoder (container restart,
no rebuild). `TEXT_ENCODER_FP8_REPO` / `TEXT_ENCODER_FP8_FILE` override the
source file. No torchao dependency.

### Sampling defaults

- **Turbo:** 8 steps, fixed by the checkpoint (see above).
- **Base model: NOT distilled** — sample at **40 steps** (the FLUX.2-klein
  default of 4 steps does not apply!). Set `NUM_INFERENCE_STEPS=40`; the
  env default is 8 and only matters for checkpoints without a fixed schedule.
- **No guidance** — the model is meant to be sampled without guidance:
  `true_cfg_scale=1.0` (`TRUE_CFG_SCALE` env var). cfg > 1 only engages
  together with a negative prompt and roughly doubles latency; the server
  does not send a negative prompt, so leave it at 1.0.
- `height`/`width` must be multiples of 32 (VAE scale factor 16 × 2);
  the server rounds request sizes to the nearest multiple of 32.
- **Native 2K** — the official default is **2048×2048** and every recommended
  aspect ratio is ~4.2 MP. Generating at 1024×1024 (the FLUX.2-klein default)
  is off-distribution for this model and yields visibly degraded, "corrupted"
  looking images. `DEFAULT_SIZE` is therefore `2048x2048`, the max edge is
  2752 and requests are area-capped at 4.4 MP.

  | Ratio | Size | Ratio | Size |
  |---|---|---|---|
  | 1:1 | 2048×2048 | 3:2 | 2528×1696 |
  | 4:3 | 2400×1792 | 2:3 | 1696×2528 |
  | 3:4 | 1792×2400 | 16:9 | 2752×1536 |
  |  |  | 9:16 | 1536×2752 |

- **Editing runs at ~1 MP**, not 2K: the official edit examples pass no
  width/height, so the pipeline derives the output from the reference image's
  aspect ratio at `output_resolution` (`EDIT_OUTPUT_RESOLUTION`, default 1024).
  This server only overrides that when the caller sends an explicit `size`.
- **RGBA output**: the VAE is a 4-channel RGBA autoencoder, so the pipeline
  returns RGBA PIL images even for ordinary opaque prompts. `ALPHA_MODE`
  (default `auto`) drops a fully-opaque alpha channel and keeps genuine
  transparency; `flatten` always composites onto white, `keep` preserves RGBA.
  For deliberate transparency use the documented prompt form: *"This is an
  RGBA image with transparency. … The image has alpha channel and the
  background is transparent."*

## Endpoints

- `GET  /health` — liveness + model id / steps / resident VRAM
- `GET  /v1/models` — OpenAI-compatible model list (one entry)
- `POST /v1/images/generations` — OpenAI-shaped JSON body:
  `{"prompt": str, "size": "1024x1024", "n": 1, "response_format": "b64_json", "steps": 40, "seed": 42}`
  (`steps` and `seed` are non-standard extras, same names as the flux2-klein
  server; Turbo ignores `steps`; `response_format` must be `b64_json`; size: max edge 2752, max 4.4 MP, i.e. every preset in the table above)
- `POST /v1/images/edits` — OpenAI-shaped multipart: `image` file +
  `prompt` form fields, plus the same optional `size`/`n`/`steps`/`seed` extras

## Build & run

Copy this whole `qwen-image-2.1-dgx-spark/` folder to the DGX Spark box, then either:

```bash
# Plain docker (mirrors deploy/dgx-spark/vllm-qwen3.6/start.sh):
cd qwen-image-2.1-dgx-spark
./start.sh          # builds the image, downloads ~25 GB of weights, waits for readiness
NO_CACHE=1 ./start.sh   # once, if the build fails its sample_sigmas check
```

```bash
# Or docker compose:
QWEN_IMAGE_PORT=8002 docker compose up -d --build
```

`start.sh` builds the image from the local Dockerfile (`nvcr.io/nvidia/pytorch:25.11-py3`
base — the only source of ARM64 + CUDA-13 PyTorch for GB10), streams
container logs to `.qwen-image.log`, and polls
`http://127.0.0.1:8002/health` until ready. First run downloads
~25 GB of weights (DiT + VAE ~15 GB, FP8 text encoder 9.4 GB; the bf16
encoder is skipped when the FP8 one is in use), so the healthcheck start-period and the poll loop are
sized generously.

### Swapping out the old FLUX.2-klein service

The old service (`deploy/dgx-spark/flux2-klein/`) uses the same host port
8002. Before starting this one, stop the old container:

```bash
# in deploy/dgx-spark/flux2-klein/ on the Spark box:
./stop.sh
```

Then start this one. The nixclaw client needs no config change
(`image_gen.base_url` already points at port 8002) — but note the latency
jump: FLUX.2-klein was ~5 s / ~9.5 s at 4 distilled steps; Turbo's 8 steps
at 2048² take roughly a minute, and the base Qwen-Image-2.1 at 40 steps
substantially longer per request (tens of seconds at
1024², more on this box than the GB300 cookbook figure of 3.3–4.8 s).
`image_gen.timeout` (180 s) and `max_concurrent_wait_seconds` (240 s) in
`config/agent.yaml` should still be safe; bump them if requests start
timing out.

## Verify

```bash
curl -s localhost:8002/health | jq
curl -s localhost:8002/v1/models | jq

# Generation:
curl -s localhost:8002/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a red bicycle leaning against a brick wall", "size": "2048x2048"}' \
  | jq -r '.data[0].b64_json' | base64 -d > /tmp/out.png

# Editing:
curl -s localhost:8002/v1/images/edits \
  -F image=@/tmp/out.png \
  -F prompt="make the bicycle blue" \
  | jq -r '.data[0].b64_json' | base64 -d > /tmp/edited.png
```

## Performance

Measured on this GB10 with `bench_backend.py --micro` (idle box):

| Measurement | Value |
|---|---|
| bf16 GEMM ceiling | **100 TFLOPS** (48 SMs) |
| fp8_e4m3 GEMM | 213 TFLOPS (**2.13×** bf16) |
| SDPA seq=16384, no mask | 90 TFLOPS (default) / 95 (cuDNN) |
| SDPA seq=16384, broadcast mask | **30 TFLOPS (default) / 85 (cuDNN)** |
| torch.compile on one block | 1.05× (not worth it; 48 SMs is below inductor's `max_autotune_gemm` threshold) |

`ATTENTION_BACKEND=_native_cudnn` is the default here, but only for the
**1.06× unmasked** gain in row 3 (46.12 vs 48.87 ms at seq=16384). It is free
and lossless, so it is worth taking — but it is not a big lever.

A plausible-looking theory that turned out to be **wrong**, recorded so nobody
re-derives it: masked SDPA is ~3× slower than unmasked (30 vs 85 TFLOPS), and
the pipeline *does* contain mask-building code, so masked attention looked like
the explanation for the model running 1.55× above its arithmetic floor. The
`ATTENTION MASK PROBE:` log line disproved it:

```
ATTENTION MASK PROBE: 32/64 calls carried a mask; shapes=[(1, 1, 16, 16)]
```

The only masked calls are **16-token** causal triangles over text segments.
The 16384-token image attention runs unmasked, because unpadded prompts leave
`encoder_hidden_states_mask` as `None`, which in turn leaves `joint_key_valid`
— and therefore the cached-decode mask — `None` as well. Measure before
optimizing; the probe stays in the code for exactly this reason.

### Remaining unexplained overhead

Per-step cost is ~1.55× the block-benchmark floor (6.46 s vs 4.18 s at
2048²), and the same ~1.6–1.9× shows up at 1024². A constant factor across
resolutions rules out attention scaling and points at per-block overhead
(unfused norm/modulation/RoPE, tensor copies, Python dispatch over 32 blocks).
To attribute it to real kernels:

```bash
python3 bench_backend.py --profile http://localhost:8002
```

Every HTTP benchmark now also writes its rendered image to
`tmp/<run-id>-<params>.png` (override with `--out-dir`), so speed numbers can
be checked against what actually came out:

```
tmp/20260922-010747-2048x2048-4steps.png
tmp/20260922-010747-1536x1536-4steps.png
```

Those benchmark renders are 4–12 steps and therefore deliberately
undersampled — they are not a quality signal. To judge a step count, use the
fixed-seed sweep, which varies **only** the step count:

```bash
python3 bench_backend.py --quality http://localhost:8002            # 40,32,24,16
python3 bench_backend.py --quality http://localhost:8002 --quality-steps 40,24
```

Pick the lowest count you cannot distinguish from 40, then set
`NUM_INFERENCE_STEPS` to it. (Base model only — Turbo's fixed schedule
ignores the step count, so this sweep renders the same 8 steps every time.)

This calls `POST /debug/profile`, which runs a short generation under
`torch.profiler` **on the already-loaded pipeline** (no extra VRAM) and
reports CUDA time split into attention / gemm / elementwise-other, plus the
top kernel table. Disable in production with `ENABLE_PROFILE_ENDPOINT=0`.

Hard floor, for expectation setting: 2048² needs **316.7 TFLOP/step**, so at
100 TFLOPS even a perfect implementation cannot beat ~3.2 s/step, i.e.
**~2.1 min for 40 steps**. Levers, in order of value:

| Lever | Gain | Cost |
|---|---|---|
| cuDNN attention (default here) | ~1.85× | none |
| `NUM_INFERENCE_STEPS` 40 → 24 | 1.7× | quality |
| FP8 DiT quantization | ~1.4× @2K | quality + work; also frees ~7 GB |
| 2048² → 1536² | 2.2× | quality (off-native) |

## Memory budget — read this before running alongside vLLM

vLLM on this box runs at `--gpu-memory-utilization 0.84`
(`deploy/dgx-spark/vllm-qwen3.6/start.sh`). Unquantized Qwen-Image-2.1's peak
memory (transformer + text encoder + VAE) is expected to be in the
~34–40 GB range; measure the real number on this box.

**Re-measure after the move to 2K.** 2048×2048 is 4× the latent tokens of
1024×1024, so attention/activation memory and step time both grow well
above whatever the first 1024² runs showed. Take the `/health`
`resident_memory_mb` reading at 2048×2048, not at 1024×1024.

**Transient spike during generation.** A 2048×2048 generation adds ~6 GB of
activation memory on top of the idle resident footprint (observed 2026-09-22:
host free RAM 24 GB → 18 GB mid-generation, released again afterward). Count
this against pool headroom, not just `resident_memory_mb` — the ~31 GB
resident figure from `/health` does not include it.

**Decode peak — the real OOM cause.** The denoising loop is the *steady* cost;
the VAE decode after the last step is the *peak*. Qwen-Image's VAE is a video
VAE (3D convs) and, untiled, decodes the whole frame at once, holding
full-resolution intermediates (~1 GB per tensor at 2048×2048 bf16, several
live at once plus conv workspace). Measured on the box 2026-09-22: host free
memory held at 18 GB through the entire ~4.5 min denoising loop, then dropped
to **6.7 GB in one steep step at decode time** (~11 GB), with the host going
unresponsive as the kernel thrashed on reclaim. That spike — not the
denoising loop — is what the OOM killer kept catching, and admission control
never saw it coming because it checks free memory *before* the request.

Tiling (`enable_vae_tiling()` + `enable_vae_slicing()`) decodes in
overlapping blended tiles instead: peak ~1–2 GB rather than ~11 GB. The cost
is **tile seams**: faint pink/purple vertical (and horizontal) stripes every
~192 px, visible in smooth low-contrast areas like fog or sky (known upstream,
[diffusers #14928](https://github.com/huggingface/diffusers/issues/14928);
same seam reported for Qwen-Image-2.1 on DGX Spark in the NVIDIA forums).
It hits generations and edits alike (every output is VAE-decoded; edits also
VAE-encode their reference images).

`VAE_TILING` picks the trade-off:

| Value | Behaviour |
|---|---|
| `auto` (default) | right before each VAE decode/encode, `empty_cache()` + measure the free pool; run **untiled** (no seams) if free ≥ `megapixels × UNTILED_VAE_MB_PER_MP` (7500, decode) or `UNTILED_VAE_ENCODE_MB_PER_MP` (2600, encode) `+ UNTILED_VAE_MARGIN_MB` (2000), else tiled. Per image (slicing). Decode at 2048² needs ~33.5 GB free, at 1024² ~9.9 GB; encoding a 1 MP reference ~4.7 GB |
| `1` / `always` | always tiled — seams, smallest peak |
| `0` / `never` | never tiled — A/B testing only. **2026-10-08: a 2048² run with ~2 GB free during denoising froze the box at decode** (pool full, 12.5 GB swapped, kernel thrashing on reclaim, ssh dead) |

**2026-10-10 freeze — the old 2700 MB/MP estimate was far too low.** A
2752×1536 (4.23 MP) *edit* finished denoising normally, then auto mode chose
an untiled decode (`28820 MB free, untiled needs 13413 MB`) and the box went
into swap (500 MB free, 3 GB swapped, `free -m` taking minutes). Seven minutes
earlier the same size decoded untiled fine as a plain generation with 31.2 GB
free. The decode code path is identical for both (only the target latent is
decoded), so the real untiled peak at 4.2 MP is close to 29–31 GB, i.e. at
least ~6.8 GB/MP. Measured with the allocator on 2026-10-11: **7183 / 7193
MB/MP** untiled (the default is now 7500), 125 MB/MP tiled. Each decode/encode
logs its real allocator peak:

```
VAE decode peak: +NNNN MB reserved, +NNNN MB allocated (untiled, 1.05 MP, NNNN MB/MP)
```

(also in `/health` `last_vae_decision.peak_reserved_mb`). Re-check
`UNTILED_VAE_MB_PER_MP` against it after diffusers or model upgrades. Note
that the free figure includes reclaimable page cache via `MemAvailable`;
reclaiming it is not free on this box, which is one more reason to keep the
margin generous.

The check runs at decode time rather than at admission because the decode
comes minutes after admission and other tenants may have grown meanwhile.
The 2.7 GB/MP figure comes from one measurement (2026-09-22, free host memory
sampled during decode, so the true peak may be a bit higher); the 2 GB margin
is deliberately narrow — raise `UNTILED_VAE_MARGIN_MB` if an untiled decode
ever pushes the box into swap. Each decision is logged (`VAE decode:
tiled/untiled (... MB free, untiled needs ... MB)`) and the last one is in
`/health` `last_vae_decision`.

"Generate small, then upscale via edit" does **not** avoid the seams: the
upscale is a new model generation whose 2K latent is decoded again, tiled
unless the pool has ~13 GB free.

Post-fix verification (2026-09-22, after deploying `VAE_TILING=1`): lowest
free memory measured during generation was ~20.6 GB vs ~24.3 GB idle — the
~6 GB transient spike, no more 6.7 GB decode cliff, no crash. Timings
(24 steps per `agent.yaml`): 512×512 edit/upscale 44.1 s; 2048×2048
text-to-image 162.9 s — matching the 24-step bench (163.1 s) almost exactly,
so the clock lock costs no throughput.

**Before running both services at once:**

1. Start `qwen-image-2.1` alone, run a generation, and record resident VRAM
   from `GET /health` (`resident_memory_mb`) or `nvidia-smi`.
2. Lower vLLM's `--gpu-memory-utilization` in
   `deploy/dgx-spark/vllm-qwen3.6/start.sh` (or whichever vLLM deploy dir is
   currently live) by enough to leave headroom for that figure, then
   restart vLLM.
3. Start both, confirm `/health` is `ok` on both `:8000` (vLLM) and
   `:8002` (qwen-image-2.1) simultaneously, and re-check vLLM decode
   throughput hasn't regressed.

This requires a vLLM restart (brief downtime) to take effect — do it in a
maintenance window, not while the agent is mid-conversation.

### OOM hardening (added after the 2026-09-21 incident)

On 2026-09-21 the box went into a **global host OOM** while three GPU tenants
shared the pool. Because GB10's 128 GB is unified, driver allocations do not
show up in any process's RSS — the kernel saw an unreclaimable machine and
killed `wireplumber`, `pipewire`, `dbus-daemon` and `systemd` *before* it got
to `sglang::scheduler` and this server. The driver had been logging
`NVRM: ... NV_ERR_NO_MEMORY` for ~6 hours beforehand.

These guards ship by default:

| Guard | Setting | Effect |
|---|---|---|
| OOM victim order | `oom_score_adj: 500` / `--oom-score-adj 500` | this container dies first, instead of dbus/systemd/the LLM |
| Admission control | size-aware, see below | a request that doesn't fit **waits** for memory, and gets a **503** naming the largest size that fits only after `ADMISSION_WAIT_SECONDS`. Free memory is the smaller of `torch.cuda.mem_get_info()` and the host's `MemAvailable`: on 2026-10-09 the CUDA number let a request through with only 3.2-4.4 GB actually available. The VAE tiling decision uses the same number |
| Restart policy | `on-failure:2` (`RESTART_POLICY` in `start.sh`) | a crashed or OOM-killed server comes back at most twice, instead of reloading ~23 GB over and over into a box that is short of memory. Not restarted after a reboot or `docker stop` |
| Fragmentation | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | one process serves 1024² … 2752×1536; varying sizes fragment the caching allocator |
| Decode peak | `VAE_TILING` (default `auto`) | untiled VAE decode only when the pool has room for it at decode time, else tiled: ~1–2 GB peak instead of ~11 GB (see "Decode peak" above) |

#### Size-aware admission (2026-10-11)

Instead of one flat floor (the old `MIN_FREE_VRAM_MB`, 8000/12000 MB, which
rejected a 512² request exactly like a 4 MP one; now ignored), each request
needs

```
n × (output_MP × DENOISE_MB_PER_MP (250) + ref_MP × REF_MB_PER_MP (2400))
  + TILED_VAE_MB (1000) + ADMISSION_MARGIN_MB (2000)
```

where `ref_MP` is ~1.05 MP per reference image (the pipeline resizes every
reference to `EDIT_OUTPUT_RESOLUTION²`). That is ~3.3 GB at 1024², ~4.0 GB at
2048², ~5.8 GB for a default 1 MP edit with one reference, ~6.6 GB for a
4.2 MP edit, ~23 GB for an edit with 8 references.

Calibration — measured with the allocator on 2026-10-11 (`Request peak` and
`VAE … peak` log lines):

| What | Measured | Setting |
|---|---|---|
| Denoising (target tokens) | whole tiled 2048² request +548 MB, 526 of it the decode → < 100 MB at 4.19 MP | `DENOISE_MB_PER_MP` 250 |
| Reference KV cache (held for the whole request) | ~2.1 GB at decode time for one 1 MP reference vs 22 MB without; theory 32 blocks × K+V × 4096 × bf16 ≈ 0.5 MB/token ≈ 2.15 GB/MP | `REF_MB_PER_MP` 2400 |
| Tiled VAE decode | 526–530 MB at 4.2 MP | `TILED_VAE_MB` 1000 |
| Untiled VAE decode | 7183 / 7193 MB/MP | `UNTILED_VAE_MB_PER_MP` 7500 |
| Untiled VAE encode | 2394 MB/MP (twice) | `UNTILED_VAE_ENCODE_MB_PER_MP` 2600 |

The denoising loop is cheap; the "~6 GB transient during a 2048² denoising
loop" recorded on 2026-09-22 was most likely the untiled decode. These peaks
count only PyTorch's allocator; host-side buffers (PIL images, PNG/base64
encoding — the same unified pool) add a few hundred MB at 4 MP, which the
margin covers.

If the pool is short, the request **waits** under the request lock, polling
every 5 s, until `ADMISSION_WAIT_SECONDS` (60) after arrival — keep that
well below the agent's 180 s HTTP timeout minus the generation itself. Only
then does it get a 503 such as:

```
insufficient GPU memory: 3500 MB free, 4049 MB required for this request (waited 60s; ...).
The largest size that fits right now is about 1408x1408 - retry with that size.
```

For edits the references' KV cache usually dominates, so a smaller output
helps little; when even 512² doesn't fit the message says to retry later.

Every request logs its real peak next to the estimate:

```
Request peak (edit): +NNNN MB reserved (admission estimate NNNN MB without margin)
```

Compare against runs whose VAE passes were tiled (an untiled decode's own
peak dominates otherwise; see its `VAE decode peak` line).

`GET /health` now reports `pool_free_mb` / `pool_total_mb` and an `admission` block (parameters plus the need at 1024² and 2048²)
so you can see the headroom the admission check is using, plus the two raw
inputs `cuda_free_mb` and `host_available_mb`.

Startup loads the pipeline with `device_map="cuda"`, straight onto the GPU.
The old `from_pretrained(...).to("cuda")` built a CPU copy first, so loading
briefly needed about twice the weights. If the installed diffusers rejects
`device_map="cuda"`, the log says so and the old path is used.

**Note:** `pipe.enable_model_cpu_offload()`, which the model card recommends
for limited memory, is *not* a fix on this hardware — host RAM and VRAM are
the same physical pool, so offloading moves bytes between accounting buckets
and frees nothing while adding latency.

## Notes

- A single `asyncio.Lock` around the pipeline call serializes requests —
  one generation/edit at a time on this GPU.
- `n` (images per request) is capped at 4 to bound one request's
  memory/time cost on a shared box.
- `size` is capped at a 2752 max edge and 4.4 MP (`MAX_SIZE`/`MAX_PIXELS`), which
  admits all official presets up to 2752×1536; rounded to a multiple of 32.
- Edit-source images are downscaled to `MAX_EDIT_LONG_EDGE`
  (env var, default `2048`) before touching the pipeline — same guard as
  the flux2-klein stack, which suffered an OOM incident from an oversized
  upload (2026-08-24).
- `torch.cuda.empty_cache()` runs after every generate/edit call so one
  large request's reserved-memory high-water mark doesn't persist and
  starve the co-located vLLM process.
- `/health`'s `resident_memory_mb` reports `memory_reserved()` (what
  `nvidia-smi` shows), not `memory_allocated()`.
- If the diffusers git-main pin or the base image's torch misbehaves,
  the failure surfaces at pipeline load — `/health` returns 503 with the
  error, and the container stays up for log inspection.

## Rollback

```bash
./stop.sh
# restore old service:
cd ../flux2-klein && ./start.sh
```
