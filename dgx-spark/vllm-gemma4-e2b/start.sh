#!/usr/bin/env bash
set -euo pipefail

# vLLM launch script for google/gemma-4-E2B-it on NVIDIA DGX Spark (GB10,
# sm_121a). Small multimodal model (text + image + audio), meant to run next
# to the main Qwen3.8 deploy (port 8000) as a fast vision / second-opinion
# STT model. See README.md for the memory budget and expected speed.
#
# Footprint estimate for WEIGHTS=bf16 (NOT measured on this box yet): weights ~10.5 GB
# (5.1B params incl. per-layer embeddings, ~150M vision + ~300M audio
# encoder), KV cache ~0.8 GB at 128K (only 3 of 35 layers hold full-context
# KV: 20 layers share KV, the rest are 512-token sliding window), plus
# ~3-4 GB CUDA context / graphs / activations => ~15 GB total.

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gemma-4-e2b}"
# Same Spark-built vLLM image as the Qwen3.8 vLLM deploys. Gemma 4 support
# in this image is assumed, not verified -- override IMAGE if it fails to
# load the architecture.
IMAGE="${IMAGE:-eugr/spark-vllm:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-gemma4-e2b}"
HOST="${HOST:-0.0.0.0}"
# 8000 = Qwen3.8, 8002/8003/8005/8008-8011 taken by other deploys.
PORT="${PORT:-8004}"

# 131072 = the model's max context.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

# bf16  = the published weights, unchanged (default).
# fp8   = vLLM online FP8 quantization of the same checkpoint at load time
#         (no separate download). Only Linear layers are quantized; the
#         per-layer embedding (PLE) tables and the tied lm_head stay BF16, so
#         this saves ~2 GB, not half.
# w4a16 = Google's official QAT checkpoint (int4 weights, group 32, BF16
#         activations, compressed-tensors). Trained for int4, so the closest
#         4-bit variant to BF16 quality. PLE, lm_head, vision and audio stay
#         BF16 (8.3 GB on disk).
# nvfp4 = unsloth's mixed NVFP4 checkpoint: FP4 W4A4 MLPs, FP8 attention and
#         sensitive MLP layers, int8 PLE table (5.5 GB on disk). Uses GB10's
#         FP4 tensor cores, but it's post-training quantization of a small model
#         (no published accuracy numbers), and unsloth requires vLLM >= 0.25.
# Decode reads the Linear weights + the BF16 lm_head per token (the PLE table
# is only a lookup), so 4-bit cuts that from ~2.7 GB (fp8) to ~1.9 GB.
WEIGHTS="${WEIGHTS:-bf16}"

# 1 = on (default), 0 = off. A disabled modality gets a limit of 0 images /
# audio clips per prompt, so vLLM skips its encoder profiling buffers; with
# both off the server runs --language-model-only and skips the encoder
# weights too. Text decode speed is the same either way.
ENABLE_VISION="${ENABLE_VISION:-1}"
ENABLE_AUDIO="${ENABLE_AUDIO:-1}"
MAX_IMAGES_PER_PROMPT="${MAX_IMAGES_PER_PROMPT:-4}"
# Each clip is capped at 30s by the model; nixclaw's STT pipeline chunks.
MAX_AUDIO_PER_PROMPT="${MAX_AUDIO_PER_PROMPT:-1}"

# MTP speculative decoding with Google's separate drafter for this model
# (158 MB BF16, 4 layers): it reuses the target's KV cache and scores ~4K
# centroid-selected candidates instead of the full 262K vocabulary, so a
# draft token costs a few percent of a target step. 2 = vLLM's Gemma 4
# recipe value for E2B (A100/H100); measure with bench.py. 0 = off. The
# drafter was trained against BF16 Gemma, so acceptance may be lower with
# the quantized checkpoints.
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-2}"
DRAFT_MODEL_ID="${DRAFT_MODEL_ID:-google/gemma-4-E2B-it-assistant}"

# Without --enable-auto-tool-choice + a parser, vLLM answers every request
# that carries "tools" with HTTP 400 ("auto" tool choice requires ...), so
# OpenAI-style tool calling doesn't work at all. Empty = off.
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-gemma4}"

# vLLM's fraction is of the WHOLE unified memory (119.63 GiB as vLLM sees it
# on GB10), not of what is free: 0.11 ~= 13.2 GiB. The default 0.9 would take
# the box down next to Qwen. vLLM gives everything in the budget beyond
# weights (~9.5 GiB BF16) and activations to the KV cache, so this is the
# hard ceiling for this server. vLLM refuses to start if util x total exceeds
# the memory free at launch (first run: 0.13 = 15.55 GiB vs 15.3 GiB free),
# and fails after loading if less than ~0.94 GiB is left for the 128K KV
# cache (w4a16 at 0.095 left 0.06 GiB, with 2.26 GiB of non-torch memory on
# top of the weights; at 0.105 it froze the box near OOM). nvfp4 runs at 0.08
# with room for ~780K KV tokens.
case "${WEIGHTS}" in
  bf16) DEFAULT_MODEL_ID=google/gemma-4-E2B-it DEFAULT_GPU_MEM_UTIL=0.11 ;;
  fp8) DEFAULT_MODEL_ID=google/gemma-4-E2B-it DEFAULT_GPU_MEM_UTIL=0.095 ;;
  w4a16) DEFAULT_MODEL_ID=google/gemma-4-E2B-it-qat-w4a16-ct DEFAULT_GPU_MEM_UTIL=0.105 ;;
  nvfp4) DEFAULT_MODEL_ID=unsloth/gemma-4-E2B-it-NVFP4 DEFAULT_GPU_MEM_UTIL=0.08 ;;
  *)
    echo "Unknown WEIGHTS='${WEIGHTS}' (expected: bf16 | fp8 | w4a16 | nvfp4)"
    exit 1
    ;;
esac
MODEL_ID="${MODEL_ID:-${DEFAULT_MODEL_ID}}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-${DEFAULT_GPU_MEM_UTIL}}"

if ! [[ "${NUM_SPECULATIVE_TOKENS}" =~ ^[0-9]+$ ]]; then
  echo "NUM_SPECULATIVE_TOKENS must be a number >= 0, got '${NUM_SPECULATIVE_TOKENS}'"
  exit 1
fi

for flag in ENABLE_VISION ENABLE_AUDIO; do
  case "${!flag}" in
    0|1) ;;
    *)
      echo "${flag} must be 0 or 1, got '${!flag}'"
      exit 1
      ;;
  esac
done

# OOM kill order on the Spark: qwen3-tts (+800), qwen-image (+500), this
# container (+300), Qwen3.8 (-900) last.
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-300}"

PID_FILE=".vllm.pid"
LOG_FILE=".vllm.log"
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_HOME="${WORK_DIR}/.cache/huggingface"
CACHE_DIR="${WORK_DIR}/.cache/vllm-runtime"
READY_URL="http://127.0.0.1:${PORT}/v1/models"

command -v docker >/dev/null 2>&1 || {
  echo "docker is not on PATH"
  exit 1
}

command -v curl >/dev/null 2>&1 || {
  echo "curl is not on PATH"
  exit 1
}

# vLLM refuses to start if util x total exceeds CUDA's free memory. On the
# Spark's unified memory CUDA does NOT count the page cache as free (it
# tracks MemFree, not MemAvailable), so a model download or any large file
# read can make the check fail although the kernel would reclaim that cache.
# Loading the weights reads ~10 GB through the cache too, which can block the
# next model start on this box. If the page cache is what stands in the way,
# this check says so; dropping it is box-wide and only costs re-reading files
# from disk: sync; echo 1 | sudo tee /proc/sys/vm/drop_caches
meminfo_kib() { awk -v k="$1:" '$1 == k {print $2}' /proc/meminfo; }

# Prints 1 if GPU_MEM_UTIL x MemTotal fits into the given /proc/meminfo field.
fits_in() {
  awk -v u="${GPU_MEM_UTIL}" -v t="$(meminfo_kib MemTotal)" -v f="$(meminfo_kib "$1")" \
    'BEGIN { print (u * t <= f) ? 1 : 0 }'
}

if [[ "$(fits_in MemFree)" == "0" && "$(fits_in MemAvailable)" == "1" ]]; then
  echo "GPU_MEM_UTIL=${GPU_MEM_UTIL} fits only once the page cache is dropped (CUDA does not count it as free)."
  echo "Run, then start again: sync; echo 1 | sudo tee /proc/sys/vm/drop_caches"
  exit 1
fi

if [[ "$(fits_in MemFree)" == "0" ]]; then
  awk -v u="${GPU_MEM_UTIL}" -v t="$(meminfo_kib MemTotal)" \
      -v f="$(meminfo_kib MemFree)" -v a="$(meminfo_kib MemAvailable)" 'BEGIN {
    printf "GPU_MEM_UTIL=%s needs %.1f GiB, but only %.1f GiB is free (%.1f GiB incl. page cache).\n", u, u * t / 1048576, f / 1048576, a / 1048576
    printf "Free memory, or try GPU_MEM_UTIL=%.3f or lower MAX_MODEL_LEN.\n", int(f / t * 1000) / 1000 - 0.005
  }'
  exit 1
fi

mkdir -p "${HF_HOME}" "${CACHE_DIR}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "Log: ${LOG_FILE}"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

VLLM_ARGS=(
  serve "${MODEL_ID}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size 1
  --dtype bfloat16
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --enable-chunked-prefill
  --enable-prefix-caching
)

if [[ -n "${TOOL_CALL_PARSER}" ]]; then
  VLLM_ARGS+=(--enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER}")
fi

if [[ "${WEIGHTS}" == "fp8" ]]; then
  VLLM_ARGS+=(--quantization fp8)
fi

if (( NUM_SPECULATIVE_TOKENS > 0 )); then
  VLLM_ARGS+=(--speculative-config "{\"model\":\"${DRAFT_MODEL_ID}\",\"num_speculative_tokens\":${NUM_SPECULATIVE_TOKENS}}")
fi

if [[ "${ENABLE_VISION}" == "0" && "${ENABLE_AUDIO}" == "0" ]]; then
  VLLM_ARGS+=(--language-model-only)
else
  image_limit=$(( ENABLE_VISION == 1 ? MAX_IMAGES_PER_PROMPT : 0 ))
  audio_limit=$(( ENABLE_AUDIO == 1 ? MAX_AUDIO_PER_PROMPT : 0 ))
  VLLM_ARGS+=(--limit-mm-per-prompt "{\"image\":${image_limit},\"audio\":${audio_limit}}")
fi

on_off() { [[ "$1" == "1" ]] && echo on || echo off; }

echo "Starting vLLM container for ${MODEL_ID}"
echo "Image: ${IMAGE}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Listening on ${HOST}:${PORT}"
echo "Weights: ${WEIGHTS}  context: ${MAX_MODEL_LEN}  gpu-memory-utilization: ${GPU_MEM_UTIL}"
echo "Vision: $(on_off "${ENABLE_VISION}")  Audio: $(on_off "${ENABLE_AUDIO}")  MTP draft tokens: ${NUM_SPECULATIVE_TOKENS}"
echo "Writing progress to ${LOG_FILE}"

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching vLLM container (weights=${WEIGHTS} vision=${ENABLE_VISION} audio=${ENABLE_AUDIO} mtp=${NUM_SPECULATIVE_TOKENS})
EOF

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  --gpus all \
  --oom-score-adj "${OOM_SCORE_ADJ}" \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e VLLM_CACHE_ROOT=/root/.cache/vllm \
  -e TRITON_CACHE_DIR=/root/.cache/triton \
  -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${CACHE_DIR}:/root/.cache" \
  --entrypoint vllm \
  "${IMAGE}" \
  "${VLLM_ARGS[@]}" \
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

echo "Waiting for HTTP readiness at ${READY_URL}"
until curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "vLLM container exited before becoming ready"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  echo "  still starting..."
  sleep 5
done

echo "vLLM is ready"
echo "OpenAI base URL: http://${HOST}:${PORT}/v1"
