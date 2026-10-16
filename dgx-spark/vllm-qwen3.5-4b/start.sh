#!/usr/bin/env bash
set -euo pipefail

# vLLM launch script for Qwen/Qwen3.5-4B on NVIDIA DGX Spark (GB10,
# sm_121a), text-only, with MTP speculative decoding. Small coding /
# tool-calling model meant to run next to the main Qwen3.8 deploy (port
# 8000). See README.md for the memory budget, the MTP depth and the
# expected speed.
#
# Architecture: 32 layers, 24 Gated DeltaNet (linear attention, fixed-size
# state per request) + 8 full-attention layers (4 KV heads x 256), so the KV
# cache is ~32 KiB per token (~4 GiB at 128K in BF16) plus the MTP layer's
# own ~4 KiB per token. 248K vocab, tied embedding / lm_head (1.27 GB BF16,
# never quantized in either checkpoint and read once per token).

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-4b}"
# Same Spark-built vLLM image as the Qwen3.8 and Gemma vLLM deploys.
IMAGE="${IMAGE:-eugr/spark-vllm:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-qwen3.5-4b}"
HOST="${HOST:-0.0.0.0}"
# 8000 = Qwen3.8, 8004 = Gemma 4 E2B, 8002/8003/8005/8008-8011 taken by
# other deploys.
PORT="${PORT:-8006}"

# 131072 to match the other deploys; the model supports 262144.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"

# fp8   = the official BF16 checkpoint with vLLM's online FP8 quantization
#         (default). Qwen publishes no FP8 checkpoint for the 4B; the
#         community ones either drop the MTP head (Shashwat42), keep the
#         DeltaNet projections in BF16 and store an untied lm_head copy
#         (RedHatAI, 8 GB), or are barely used (Hob-forge, block FP8 in the
#         official format: MODEL_ID=Hob-forge/Qwen3.5-4B-FP8).
#         Per token ~3.6 GB FP8 Linear + 1.27 GB BF16 lm_head.
# nvfp4 = AxionML's ModelOpt NVFP4 checkpoint (W4A4, group 16, 4.2 GB):
#         keeps the MTP head (BF16), lm_head and embeddings in BF16, ~1-3%
#         accuracy drop per the card. cosmicproc/Qwen3.5-4B-NVFP4 is the
#         same quantization WITHOUT the MTP head -- don't use it with MTP.
#         Per token ~2.0 GB FP4 Linear + 1.27 GB BF16 lm_head.
WEIGHTS="${WEIGHTS:-fp8}"

# MTP: Qwen3.5 ships one multi-token-prediction layer in the checkpoint; it
# drafts NUM_SPECULATIVE_TOKENS tokens per step, the full model verifies
# them in one pass. Each draft token costs one MTP layer + one pass over the
# 1.27 GB lm_head (~30% of an FP8 step, ~45% of an NVFP4 step), so deep
# drafts don't pay off on a 4B model. 2 = estimated optimum for fp8, 1-2
# for nvfp4 (see README "MTP depth"); measure with bench.py. 0 = off.
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-2}"

# 1 = the model's default (thinks before answering; the published
# LiveCodeBench / BFCL numbers are in this mode), 0 = answer directly.
# Clients can override per request with
# chat_template_kwargs: {"enable_thinking": ...}.
THINKING="${THINKING:-1}"

# auto = BF16 KV cache. fp8 halves it (~2 GiB at 128K instead of ~4 GiB)
# without calibrated scales; untested with this model on this box.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"

# Without --enable-auto-tool-choice + a parser, vLLM answers every request
# that carries "tools" with HTTP 400. qwen3_coder = the parser vLLM's Qwen3.5
# recipe names. Empty = off.
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"

# vLLM's fraction is of the WHOLE unified memory (119.63 GiB as vLLM sees it
# on GB10), not of what is free. Estimates, not measured yet: weights ~5.3
# GiB (fp8) / ~3.5 GiB (nvfp4) incl. MTP, ~2.3 GiB non-torch (CUDA context,
# workspaces, as measured for Gemma), ~1.5 GiB activations, ~4.7 GiB KV for
# one 128K request incl. the MTP layer => 0.12 = 14.4 GiB for fp8 (starts).
# nvfp4 at 0.105 (12.6 GiB) left 4.31 GiB for the 4.73 GiB a 128K request
# needs (vLLM also pads the KV layers by up to 12.5%), so it gets 0.112
# (13.4 GiB): the 4-bit weights save ~1.8 GB, not the whole difference.
# vLLM stops after loading if less than one max-length request's KV fits
# ("estimated maximum model length is ..."): raise by 0.01 or lower
# MAX_MODEL_LEN / set KV_CACHE_DTYPE=fp8.
case "${WEIGHTS}" in
  fp8) DEFAULT_MODEL_ID=Qwen/Qwen3.5-4B DEFAULT_GPU_MEM_UTIL=0.12 ;;
  nvfp4) DEFAULT_MODEL_ID=AxionML/Qwen3.5-4B-NVFP4 DEFAULT_GPU_MEM_UTIL=0.112 ;;
  *)
    echo "Unknown WEIGHTS='${WEIGHTS}' (expected: fp8 | nvfp4)"
    exit 1
    ;;
esac
MODEL_ID="${MODEL_ID:-${DEFAULT_MODEL_ID}}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-${DEFAULT_GPU_MEM_UTIL}}"

case "${THINKING}" in
  0|1) ;;
  *)
    echo "THINKING must be 0 or 1, got '${THINKING}'"
    exit 1
    ;;
esac

if ! [[ "${NUM_SPECULATIVE_TOKENS}" =~ ^[0-9]+$ ]]; then
  echo "NUM_SPECULATIVE_TOKENS must be a number >= 0, got '${NUM_SPECULATIVE_TOKENS}'"
  exit 1
fi

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
# If the page cache is what stands in the way, this check says so; dropping
# it is box-wide and only costs re-reading files from disk:
# sync; echo 1 | sudo tee /proc/sys/vm/drop_caches
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
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --kv-cache-dtype "${KV_CACHE_DTYPE}"
  --enable-chunked-prefill
  --enable-prefix-caching
  # Skips the vision encoder's weights and profiling buffers.
  --language-model-only
  --reasoning-parser qwen3
)

if [[ -n "${TOOL_CALL_PARSER}" ]]; then
  VLLM_ARGS+=(--enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER}")
fi

# The NVFP4 checkpoint's quantization is auto-detected (hf_quant_config.json).
if [[ "${WEIGHTS}" == "fp8" ]]; then
  VLLM_ARGS+=(--quantization fp8)
fi

if [[ "${THINKING}" == "0" ]]; then
  VLLM_ARGS+=(--default-chat-template-kwargs '{"enable_thinking":false}')
fi

if (( NUM_SPECULATIVE_TOKENS > 0 )); then
  VLLM_ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPECULATIVE_TOKENS}}")
fi

echo "Starting vLLM container for ${MODEL_ID}"
echo "Image: ${IMAGE}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Listening on ${HOST}:${PORT}"
echo "Weights: ${WEIGHTS}  context: ${MAX_MODEL_LEN}  gpu-memory-utilization: ${GPU_MEM_UTIL}  kv-cache: ${KV_CACHE_DTYPE}"
echo "MTP draft tokens: ${NUM_SPECULATIVE_TOKENS}  thinking: ${THINKING}"
echo "Writing progress to ${LOG_FILE}"

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching vLLM container (weights=${WEIGHTS} mtp=${NUM_SPECULATIVE_TOKENS} thinking=${THINKING})
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
