#!/usr/bin/env bash
set -euo pipefail

# SGLang launch script for LiquidAI/LFM3-1.7B-NVFP4 on NVIDIA DGX Spark (GB10,
# sm_121a, native FP4 tensor cores). See README.md.
#
# NVFP4 (ModelOpt W4A4, FFN only; attention/conv/embeddings stay BF16) is the
# default weight format here. It can only be served by SGLang, and only with
# the LFM3 patch from sgl-project/sglang#46858 (open, unmerged as of writing),
# so this deploy builds a thin local image (./Dockerfile) that overlays the PR
# onto lmsysorg/sglang:dev-cu13. UNVERIFIED on the box until first run.
# Fallback with the same model on stock images: ../llamacpp-lfm3-1.7b (GGUF).
#
# Port 8010 by default. NOTE: ../qwen3-tts also defaults to 8010 -- start.sh
# refuses to start if the port is bound; override with PORT=<n>.

MODEL_ID="${MODEL_ID:-LiquidAI/LFM3-1.7B-NVFP4}"
# Provenance only (revs are not passed to --model-path, see ../lfm2.5).
MODEL_REV="${MODEL_REV:-main}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-lfm3-1.7b}"
BASE_IMAGE="${BASE_IMAGE:-lmsysorg/sglang:dev-cu13}"
IMAGE="${IMAGE:-nixclaw/sglang-lfm3:local}"
REBUILD="${REBUILD:-0}"
LFM3_PR="${LFM3_PR:-46858}"
CONTAINER_NAME="${CONTAINER_NAME:-sglang-lfm3-1.7b-nvfp4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8010}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-131072}"
# A 1.7B model needs a few GB, not the 0.80 the big hybrid deploys use.
# Keep the KV pool small so this coexists with the other Spark deploys
# (qwen on 8000, gemma/qwen3.5 on 8004/8006) on the shared 128GB. Raise only
# if you want more concurrent long contexts.
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.15}"
# Empty = let SGLang read hf_quant_config.json (model card passes no flag).
# If it does not pick FP4 up: QUANTIZATION=modelopt_fp4 ./start.sh
QUANTIZATION="${QUANTIZATION:-}"
GPU_DOCKER_ARG="${GPU_DOCKER_ARG:---gpus all}"

PID_FILE=".sglang.pid"
LOG_FILE=".sglang.log"
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_HOME="${WORK_DIR}/.cache/huggingface"
CACHE_DIR="${WORK_DIR}/.cache/sglang-runtime"
PR_DIFF="${WORK_DIR}/lfm3-sglang-pr.diff"
READY_URL="http://127.0.0.1:${PORT}/health"

cd "${WORK_DIR}"
command -v docker >/dev/null 2>&1 || { echo "docker is not on PATH"; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl is not on PATH"; exit 1; }
mkdir -p "${HF_HOME}" "${CACHE_DIR}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "Log: ${LOG_FILE}"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

if command -v ss >/dev/null 2>&1 && ss -H -ltn "sport = :${PORT}" | grep -q .; then
  echo "Port ${PORT} is already in use on this host:"
  ss -H -ltnp "sport = :${PORT}" || true
  echo "Stop the other service or pick another port: PORT=<n> ./start.sh"
  exit 1
fi

# Fetch the PR diff once; it stays pinned in this directory afterwards.
# Delete ./lfm3-sglang-pr.diff and REBUILD=1 to pick up PR updates.
if [[ ! -s "${PR_DIFF}" ]]; then
  echo "Downloading sgl-project/sglang#${LFM3_PR} diff"
  curl -fsSL "https://github.com/sgl-project/sglang/pull/${LFM3_PR}.diff" -o "${PR_DIFF}.part"
  mv "${PR_DIFF}.part" "${PR_DIFF}"
fi

if [[ "${REBUILD}" == "1" ]] || ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Building ${IMAGE} from ${BASE_IMAGE} + PR #${LFM3_PR} (fails loudly if the patch does not apply)"
  docker build --build-arg "BASE_IMAGE=${BASE_IMAGE}" -t "${IMAGE}" "${WORK_DIR}"
fi

SGLANG_ARGS=(
  -m sglang.launch_server
  --trust-remote-code
  --model-path "${MODEL_ID}"
  --tp 1
  --served-model-name "${SERVED_MODEL_NAME}"
  --context-length "${CONTEXT_LENGTH}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --reasoning-parser auto
  --tool-call-parser auto
  --host "${HOST}"
  --port "${PORT}"
)
if [[ -n "${QUANTIZATION}" ]]; then
  SGLANG_ARGS+=(--quantization "${QUANTIZATION}")
fi

echo "Starting SGLang container for ${MODEL_ID} (NVFP4)"
echo "Image: ${IMAGE}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Listening on ${HOST}:${PORT}, context ${CONTEXT_LENGTH}, mem-fraction-static ${MEM_FRACTION_STATIC}"
echo "Writing progress to ${LOG_FILE}"

cat >"${LOG_FILE}" <<LOGEOF
[$(date -Is)] launching SGLang container (${MODEL_ID}, PR #${LFM3_PR} overlay)
LOGEOF

docker run -d \
  --name "${CONTAINER_NAME}" \
  ${GPU_DOCKER_ARG} \
  --memory 32g \
  --memory-swap 32g \
  --shm-size 8g \
  --network host \
  --ipc host \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  -v "${CACHE_DIR}:/root/.cache" \
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

echo "Waiting for HTTP readiness at ${READY_URL} (first start downloads ~2GB model)"
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
echo "Set config/agent.yaml: backend: \"vllm\", vllm.host: \"http://<spark>:${PORT}\", vllm.model: \"${SERVED_MODEL_NAME}\""
