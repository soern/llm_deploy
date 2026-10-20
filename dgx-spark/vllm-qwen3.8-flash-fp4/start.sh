#!/usr/bin/env bash
set -euo pipefail

# vLLM launch script for nvidia/Qwen3.8-Flash-Next-FP4-Local on NVIDIA DGX
# Spark (GB10, sm_121a), sized to share the box with ../qwen-image-2.1
# (~30 GB). See README.md for the memory budget and the reasoning behind
# each default.
#
# Base recipe: the model card's DGX Spark section
# https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local
# The checkpoint's NVFP4 scale factors use a new compression format that
# upstream vLLM can't load yet, so the card's community image is required.
#
# Differences from the card's recipe (all to fit next to the image model):
#   --gpu-memory-utilization 0.69 (card: 0.8)
#   --max-num-seqs 2              (card: 8)
#   --max-num-batched-tokens 2048 (card: 4096)
#   video input off, image input kept, multimodal processor cache off
#   plus a start-up check that refuses to launch without enough free memory.

MODEL_ID="${MODEL_ID:-nvidia/Qwen3.8-Flash-Next-FP4-Local}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-flash}"
IMAGE="${IMAGE:-eugr/spark-vllm-b12x:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-qwen3.8-flash-fp4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Memory, measured on the Spark 2026-10-20 (KV cache pool size):
#   0.65 -> fails (0.47 GiB KV, less than one 131K request)
#   0.68 -> 194K tokens
#   0.69 -> 297K tokens (default: full 262K context, ~4.6 GiB left
#           with qwen-image-2.1 idle)
#   0.70 -> 396K tokens (only ~2.3 GiB left with qwen-image-2.1: too tight)
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.69}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"

# MTP draft head ships in the checkpoint. 4 draft tokens is the card's
# value; 3 is worth trying for mostly-prose traffic.
ENABLE_MTP="${ENABLE_MTP:-1}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-4}"

# Vision stays on by default (image input). Video is off: vLLM reserves
# memory for the largest allowed video at startup. LANGUAGE_ONLY=1 drops
# the vision encoder entirely and saves another ~1-3 GiB.
LANGUAGE_ONLY="${LANGUAGE_ONLY:-0}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\":2,\"video\":0}}"

TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"

# Start-up memory check: vLLM briefly needs ~10 GiB more while loading
# than at steady state. Refuse to start unless MemAvailable covers
# GPU_MEM_UTIL x MemTotal + LOAD_HEADROOM_GIB. FORCE=1 skips the check.
LOAD_HEADROOM_GIB="${LOAD_HEADROOM_GIB:-10}"
FORCE="${FORCE:-0}"

PID_FILE=".vllm.pid"
LOG_FILE=".vllm.log"
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Keep these on the Spark's local NVMe: the n-gram table (~27 GiB) is read
# from disk on every decode step.
HF_HOME="${HF_HOME_DIR:-${WORK_DIR}/.cache/huggingface}"
CACHE_DIR="${CACHE_DIR:-${WORK_DIR}/.cache/runtime}"
READY_URL="http://127.0.0.1:${PORT}/v1/models"

cd "${WORK_DIR}"

for cmd in docker curl awk; do
  command -v "${cmd}" >/dev/null 2>&1 || {
    echo "${cmd} is not on PATH"
    exit 1
  }
done

mkdir -p "${HF_HOME}" "${CACHE_DIR}"/{vllm,flashinfer,triton,b12x}

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

if [[ "${FORCE}" != "1" ]]; then
  read -r mem_total_kib mem_avail_kib < <(
    awk '/^MemTotal:/ {t=$2} /^MemAvailable:/ {a=$2} END {print t, a}' /proc/meminfo
  )
  need_gib="$(awk -v t="${mem_total_kib}" -v u="${GPU_MEM_UTIL}" -v h="${LOAD_HEADROOM_GIB}" \
    'BEGIN {printf "%.1f", t * u / 1048576 + h}')"
  avail_gib="$(awk -v a="${mem_avail_kib}" 'BEGIN {printf "%.1f", a / 1048576}')"
  if awk -v a="${avail_gib}" -v n="${need_gib}" 'BEGIN {exit !(a < n)}'; then
    echo "Not enough free memory: ${avail_gib} GiB available, ~${need_gib} GiB needed"
    echo "(GPU_MEM_UTIL=${GPU_MEM_UTIL} of MemTotal + ${LOAD_HEADROOM_GIB} GiB load headroom)."
    echo "Start this before ../qwen-image-2.1, or stop the image model first."
    echo "FORCE=1 skips this check."
    exit 1
  fi
  echo "Memory check OK: ${avail_gib} GiB available, ~${need_gib} GiB needed"
fi

VLLM_ARGS=(
  serve "${MODEL_ID}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size 1
  --pipeline-parallel-size 1
  --quantization modelopt_mixed
  --load-format b12x
  --dtype bfloat16
  --kv-cache-dtype fp8
  --block-size 16
  --mamba-cache-mode align
  --enable-prefix-caching
  --enable-chunked-prefill
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --gdn-decode-kernel b12x
  --linear-backend b12x
  --moe-backend b12x
  --no-enable-flashinfer-autotune
  --compilation-config '{"pass_config":{"fuse_act_quant":true}}'
  --reasoning-parser qwen3
  --tool-call-parser "${TOOL_CALL_PARSER}"
  --enable-auto-tool-choice
  --generation-config vllm
  --chat-template /workspace/chat_template.jinja
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}'
)

if [[ "${LANGUAGE_ONLY}" == "1" ]]; then
  VLLM_ARGS+=(--language-model-only)
else
  VLLM_ARGS+=(
    --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}"
    --mm-encoder-tp-mode data
    # Default is a 4 GiB CPU-side cache, which on GB10 comes out of the
    # same unified memory as everything else.
    --mm-processor-cache-gb 0
  )
fi

if [[ "${ENABLE_MTP}" == "1" ]]; then
  VLLM_ARGS+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPECULATIVE_TOKENS}}")
fi

echo "Starting vLLM container for ${MODEL_ID}"
echo "Image: ${IMAGE}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Listening on ${HOST}:${PORT}"
echo "Vision: $([ "${LANGUAGE_ONLY}" = "1" ] && echo off || echo "on ${LIMIT_MM_PER_PROMPT}")  MTP: ${ENABLE_MTP} (k=${NUM_SPECULATIVE_TOKENS})"
echo "Memory: util=${GPU_MEM_UTIL} max-model-len=${MAX_MODEL_LEN} max-num-seqs=${MAX_NUM_SEQS}"
echo "Writing progress to ${LOG_FILE}"

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching vLLM container (Qwen3.8-Flash-Next FP4-Local)
EOF

# seccomp/apparmor unconfined: from the model card; Docker's default
# seccomp profile blocks io_uring, which the disk-backed n-gram table uses.
docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  --gpus all \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e VLLM_PLE_TABLE_MEMORY=disk \
  -e B12X_DISK_BACKEND=io_uring \
  -e CUTE_DSL_ARCH=sm_121a \
  -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_SSM_CONV_STATE_LAYOUT=DS \
  -e VLLM_USE_AOT_COMPILE=1 \
  -e VLLM_USE_MEGA_AOT_ARTIFACT=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e B12X_POLICY_MODE=auto \
  -e VLLM_MXFP8_LM_HEAD=1 \
  -e VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13 \
  -e VLLM_B12X_MOE_FP4_FORCE_A16=1 \
  -e B12X_W4A16_A4_PREFILL=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${CACHE_DIR}/vllm:/root/.cache/vllm" \
  -v "${CACHE_DIR}/flashinfer:/root/.cache/flashinfer" \
  -v "${CACHE_DIR}/triton:/root/.triton" \
  -v "${CACHE_DIR}/b12x:/root/.cache/b12x" \
  -v "${WORK_DIR}/chat_template.jinja:/workspace/chat_template.jinja:ro" \
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

# First start downloads ~96 GiB and compiles kernels; expect 10-20+ min.
echo "Waiting for HTTP readiness at ${READY_URL}"
until curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "vLLM container exited before becoming ready"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  echo "  still starting..."
  sleep 10
done

echo "vLLM is ready"
grep -E "KV cache size|Maximum concurrency" "${LOG_FILE}" | tail -n 2 || true
echo "OpenAI base URL: http://${HOST}:${PORT}/v1"
echo "Start ../qwen-image-2.1 now, not before."
