#!/usr/bin/env bash
set -euo pipefail

# SGLang launch script for nvidia/Qwen3.8-27B-NVFP4 + DFlash2 speculative
# decoding on NVIDIA DGX Spark (GB10, sm_121a). Optimized for 4 concurrent
# users. See README.md for the full rationale, sourcing, and trust notes.
#
# Flags validated for 4-concurrent-user workload via community testing
# (NVIDIA Developer Forums #380257, post #95) and official SGLang cookbook
# (docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B). This is a
# plain `docker run` against the official upstream image, same pattern as
# our other deploy/ scripts — no custom image build, no sudo.
#
# Performance targets: 31-36 tok/s mean at 4 concurrent (vs. ~10-30 tok/s
# single-stream on the older DSpark config). See README.md "Tuning".

# NVIDIA's ModelOpt NVFP4 checkpoint (NVFP4 MLPs, FP8 attention projections).
# Measured on this box vs. Qwen/Qwen3.8-27B-FP8: +35% decode, same tool-call
# accuracy (README "Measured on this box"). Alternatives:
# RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead, Qwen/Qwen3.8-27B-FP8.
MODEL_ID="${MODEL_ID:-nvidia/Qwen3.8-27B-NVFP4}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"

# DFlash2 draft model (speculative decoding) - no version override needed
DRAFT_MODEL_ID="${DRAFT_MODEL_ID:-incoai/Qwen3.8-27B-DFlash2}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8}"
# Official SGLang-team image, built and validated for this exact model
# (all 48 checkpoint x speculative x strategy x dtype combos re-verified on
# GB10/aarch64 at commit 1cf2b8c) -- supports both DSpark and DFlash2 without
# custom build. Moving :dev-* tag, not a pinned digest -- SGLang doesn't
# publish a digest-pinned release of this dev image; pull once and keep
# using the local copy if you want reproducibility across restarts.
IMAGE="${IMAGE:-lmsysorg/sglang:dev-qwen38-27b-dflash2}"
CONTAINER_NAME="${CONTAINER_NAME:-sglang-qwen3.8-dspark}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# DGX Spark's 128GB is unified memory shared with the host CPU. Tuned for
# 4-concurrent-user workload (NVIDIA forums post #95): 0.60 provides stable
# operation at full concurrency with headroom for memory spikes during prefill.
# Official cookbook validated 0.80 as ceiling at concurrency 1, but community
# testing found 0.60 safer at 4 concurrent (leaves ~51GB free for OS + spikes).
# Still not the same accounting mechanism as vLLM's --gpu-memory-utilization
# (see the sibling vllm-qwen3.8-nvfp4-dgx-spark deploy) -- don't equate the
# two engines' memory knobs.
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.65}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-4}"

# Explicit cache sizes, so SGLang's footprint does not depend on what else is
# loaded at startup: without --max-total-tokens the KV pool is sized from the
# memory free at launch (671k tokens / 26.9 GB on an empty box, much less if
# qwen-image started first). Measured 2026-10-09 with Qwen3.8-27B-FP8 + DFlash2:
# ~40 KB/token (main + draft KV, fp8_e4m3) and ~127 MB per mamba slot. Peak use
# the night before was 10% of 671k tokens and 12 of 96 mamba slots.
# 262144 = the model's max context, so no single request can exceed the pool
# (a second request that doesn't fit waits in SGLang's queue). Each running
# request takes 3 mamba slots, so 32 covers MAX_RUNNING_REQUESTS=4 plus prefix
# cache. MEM_FRACTION_STATIC stays as the ceiling.
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-262144}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-32}"

# OOM kill order on the Spark: qwen3-tts (+800) first, then qwen-image (+500),
# this container last. GPU allocations don't count toward a process's OOM
# score on unified memory, so without this the scheduler (~4 GB of CPU-side
# memory) would be the biggest target. -900, not -1000: a runaway sglang can
# still be killed instead of freezing the box. earlyoom uses the same score.
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:--900}"
# No Docker restart policy: watchdog.sh (sglang-watchdog.service) restarts it
# after a crash or hang, but only once there is memory for it.

# FlashInfer autotune: disabled by default for stable 4-concurrent operation.
# While autotune measured +41-54% decode speed at concurrency 1 on this box,
# community testing found it introduces variance and occasional stalls at
# concurrency 4 (NVIDIA forums post #95). Deterministic throughput 31-36 tok/s
# is more valuable than peak-but-flaky performance. Set FLASHINFER_AUTOTUNE=1
# if your workload is mostly single-user and you want the peak-speed speedup.
# Toggling this re-runs the autotune pass on next boot, adding to startup.
FLASHINFER_AUTOTUNE="${FLASHINFER_AUTOTUNE:-0}"

# Validated on this box (with FLASHINFER_AUTOTUNE=1): float32 was flat on
# code/free-prose and 15% WORSE on reasoning vs. bfloat16, while costing 2x
# the mamba state-slot memory (154MB vs 78MB/slot) -- pure downside here.
# Stays bfloat16; see README.md "Tuning" before re-testing float32.
MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-bfloat16}"

# The official cookbook notes DGX Spark's docker GPU access is CDI-only
# (no `nvidia` runtime registered there) -- `--device nvidia.com/gpu=all`
# instead of `--gpus all`. Our sibling vllm-qwen3.8-nvfp4-dgx-spark deploy
# uses `--gpus all` successfully on this box, so that stays the default
# here too; if this container can't see the GPU, set GPU_DOCKER_ARG
# yourself, e.g.: GPU_DOCKER_ARG="--device nvidia.com/gpu=all" ./start.sh
GPU_DOCKER_ARG="${GPU_DOCKER_ARG:---gpus all}"

# Speculative decoding mode: dflash2 (default, separate draft model) or mtp
# (multi-token prediction head trained into the target checkpoint -- no draft
# model download, no draft weights/KV/CUDA graphs, roughly 10-12 GB less memory
# on this box). Community benchmarks (BENCHMARK_REPORT.md) put MTP ~1.5-1.9x
# slower than DFlash2 on this hardware; use it to trade speed for headroom.
# MTP uses SGLang's NEXTN algorithm; steps/draft-tokens are tunable below.
SPEC_ALGO="${SPEC_ALGO:-dflash2}"
MTP_NUM_STEPS="${MTP_NUM_STEPS:-3}"
MTP_NUM_DRAFT_TOKENS="${MTP_NUM_DRAFT_TOKENS:-4}"

case "${SPEC_ALGO}" in
  dflash2|mtp) ;;
  *)
    echo "Unknown SPEC_ALGO='${SPEC_ALGO}' (expected: dflash2 | mtp)"
    exit 1
    ;;
esac

PID_FILE=".sglang.pid"
LOG_FILE=".sglang.log"
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_HOME="${WORK_DIR}/.cache/huggingface"
CACHE_DIR="${WORK_DIR}/.cache/sglang-runtime"
READY_URL="http://127.0.0.1:${PORT}/health"

command -v docker >/dev/null 2>&1 || {
  echo "docker is not on PATH"
  exit 1
}

command -v curl >/dev/null 2>&1 || {
  echo "curl is not on PATH"
  exit 1
}

# JIT/compile caches that would otherwise live in the container's writable
# layer and vanish on every `docker rm`: FlashInfer kernels (~/.cache/flashinfer,
# nvcc-built for sm_121a), Triton kernels (TRITON_CACHE_DIR), plus anything
# else that writes under ~/.cache (DeepGEMM, torch extensions, ...). Mounting
# the whole ~/.cache catches those without tracking each tool's env var; the
# huggingface bind mount below is nested on top of it and wins for that path.
HOME_CACHE_DIR="${CACHE_DIR}/home-cache"

mkdir -p "${HF_HOME}" "${CACHE_DIR}" "${HOME_CACHE_DIR}" "${CACHE_DIR}/triton"

if [[ ! -f "${WORK_DIR}/chat_template.jinja" ]]; then
  echo "Missing chat_template.jinja in ${WORK_DIR}"
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "Log: ${LOG_FILE}"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

SGLANG_ARGS=(
  -m sglang.launch_server
  --trust-remote-code
  --model-path "${MODEL_ID}"
  --tp-size 1
  --served-model-name "${SERVED_MODEL_NAME}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --attention-backend flashinfer
  --chunked-prefill-size 8192
  --disable-prefill-cuda-graph
  --cuda-graph-max-bs 8
  --mamba-radix-cache-strategy extra_buffer_lazy
  --mamba-ssm-dtype "${MAMBA_SSM_DTYPE}"
  --max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE}"
  --max-total-tokens "${MAX_TOTAL_TOKENS}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --kv-cache-dtype "${KV_CACHE_DTYPE}"
  --enable-torch-compile
  --torch-compile-max-bs 4
  --num-continuous-decode-steps 2
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder
  --chat-template /workspace/chat_template.jinja
  --host "${HOST}"
  --port "${PORT}"
)

if [[ "${SPEC_ALGO}" == "mtp" ]]; then
  SGLANG_ARGS+=(
    --speculative-algorithm NEXTN
    --speculative-num-steps "${MTP_NUM_STEPS}"
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens "${MTP_NUM_DRAFT_TOKENS}"
  )
  SPEC_LABEL="MTP (NEXTN, steps=${MTP_NUM_STEPS}, draft-tokens=${MTP_NUM_DRAFT_TOKENS})"
else
  SGLANG_ARGS+=(
    --speculative-algorithm DFLASH
    --speculative-draft-model-path "${DRAFT_MODEL_ID}"
    --speculative-num-draft-tokens 8
  )
  SPEC_LABEL="DFlash2 (draft: ${DRAFT_MODEL_ID})"
fi

if [[ "${FLASHINFER_AUTOTUNE}" != "1" ]]; then
  SGLANG_ARGS+=(--disable-flashinfer-autotune)
fi

echo "Starting SGLang container for ${MODEL_ID} (NVFP4, 4-concurrent optimized)"
echo "Image: ${IMAGE}"
echo "Speculative decoding: ${SPEC_LABEL}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Listening on ${HOST}:${PORT}"
echo "mem-fraction-static: ${MEM_FRACTION_STATIC}  max-running-requests: ${MAX_RUNNING_REQUESTS}  max-total-tokens: ${MAX_TOTAL_TOKENS}  max-mamba-cache-size: ${MAX_MAMBA_CACHE_SIZE}"
echo "flashinfer-autotune: ${FLASHINFER_AUTOTUNE}  chunked-prefill-size: 8192"
echo "Writing progress to ${LOG_FILE}"

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching SGLang container (NVFP4 + ${SPEC_LABEL}, optimized for 4 concurrent users)
EOF

docker run -d \
  --name "${CONTAINER_NAME}" \
  ${GPU_DOCKER_ARG} \
  --oom-score-adj "${OOM_SCORE_ADJ}" \
  --restart no \
  --memory 100g \
  --memory-swap 100g \
  --shm-size 16g \
  --network host \
  --ipc host \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
  -e TRITON_CACHE_DIR=/cache/triton \
  -v "${HOME_CACHE_DIR}:/root/.cache" \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${CACHE_DIR}:/cache" \
  -v "${WORK_DIR}/chat_template.jinja:/workspace/chat_template.jinja" \
  --entrypoint python3 \
  "${IMAGE}" \
  "${SGLANG_ARGS[@]}" \
  >/dev/null

container_id="$(docker inspect -f '{{.Id}}' "${CONTAINER_NAME}")"
echo "${container_id}" > "${PID_FILE}"
echo "Spawned container ${CONTAINER_NAME} (${container_id})"

log_follow_pid=""
cleanup() {
  if [[ -n "${log_follow_pid}" ]] && kill -0 "${log_follow_pid}" 2>/dev/null; then
    kill "${log_follow_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

(docker logs -f "${CONTAINER_NAME}" >> "${LOG_FILE}" 2>&1) &
log_follow_pid=$!

echo "Waiting for HTTP readiness at ${READY_URL} (first boot ~7-9min: torch.compile + CUDA graph capture)"
until curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "SGLang container exited before becoming ready"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  echo "  still starting..."
  sleep 5
done

echo "SGLang is ready"
echo "OpenAI base URL: http://${HOST}:${PORT}/v1"
