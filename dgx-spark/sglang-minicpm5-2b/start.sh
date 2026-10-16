#!/usr/bin/env bash
set -euo pipefail

# SGLang launch script for openbmb/MiniCPM5-2B (native BF16, 2.5B dense
# LlamaForCausalLM) on NVIDIA DGX Spark (GB10, sm_121a). See README.md for
# the full rationale and sources.
#
# SGLang (not vLLM) because the model card and SGLang's official cookbook
# recommend SGLang for this model's tool calling: MiniCPM5-2B emits
# XML-style <function ...><param ...> tool calls, and SGLang ships a built-in
# `minicpm5` detector that converts them to OpenAI-compatible `tool_calls`;
# vLLM has no equivalent parser as of this writing. nixclaw's VLLMClient
# speaks the same OpenAI-compatible /v1/chat/completions API SGLang serves.
#
# Flags follow SGLang's official cookbook cell for this exact model +
# hardware (docs.sglang.io/cookbook/autoregressive/OpenBMB/MiniCPM5-2B,
# config source docs/src/snippets/configs/openbmb/minicpm5-2b.jsx, verified
# on DGX Spark at sglang dev @ 30705c004c):
#   --mem-fraction-static 0.30     REQUIRED on GB10, not a tuning value --
#                                  the default fraction claims ~89GB of the
#                                  128GB unified memory for the KV pool,
#                                  leaving ~5GB for the OS, which kills the
#                                  node during warmup with no traceback.
#   --cuda-graph-max-bs-decode 128 from the same verified cell
#   --reasoning-parser qwen3       the chat template is Qwen-style
#                                  (<system> + <think>); there is no
#                                  minicpm5 reasoning parser
#   --tool-call-parser minicpm5    see above
# Optional DSpark speculative decoding (openbmb/MiniCPM5-2B-DSpark, ~648MB
# draft checkpoint) is the model card's own recommended overlay; off by
# default because no DSpark speed numbers are published for any card yet --
# set ENABLE_DSPARK=1 to try it.
#
# Plain `docker run` against the official upstream image, same pattern as
# the sibling ../sglang-qwen3.8 deploy -- no custom image build,
# no sudo.

MODEL_ID="${MODEL_ID:-openbmb/MiniCPM5-2B}"
# Provenance only (same convention as ../sglang-qwen3.8): the revs
# below are NOT passed to --model-path. This image's server args hand the
# string straight to transformers, whose validate_repo_id rejects '@' -- an
# "repo@revision" path dies at startup with an HFValidationError.
MODEL_REV="${MODEL_REV:-12a3808a956f869c767195e9266b59c4d21d92e2}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-minicpm5-2b}"
IMAGE="${IMAGE:-lmsysorg/sglang:dev}"
CONTAINER_NAME="${CONTAINER_NAME:-sglang-minicpm5-2b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Verified DGX Spark value from the cookbook cell (see header) -- do NOT
# raise toward the sibling qwen3.8 deploy's 0.80: that box's accounting
# (dedicated-VRAM-style reporting under a cgroup cap, hybrid-mamba model)
# does not apply here. On a pure unified-memory GB10 the cookbook's default
# OOM-kills the OS during warmup.
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.30}"
CUDA_GRAPH_MAX_BS_DECODE="${CUDA_GRAPH_MAX_BS_DECODE:-128}"

# DSpark speculative decoding: a separately published BF16 draft checkpoint
# (~648MB). Off by default -- the cookbook publishes no DSpark speed numbers
# on any card yet ("the speedup tracks acceptance length, which moves with
# the prompt distribution"). The base checkpoint is plain Llama and does not
# need --trust-remote-code; the draft checkpoint's config does, so the flag
# is only added with the overlay.
ENABLE_DSPARK="${ENABLE_DSPARK:-0}"
DRAFT_MODEL_ID="${DRAFT_MODEL_ID:-openbmb/MiniCPM5-2B-DSpark}"
DRAFT_REV="${DRAFT_REV:-114a20fdbf53220712c7fbdd7dccddbf1dedebb4}"
DSPARK_BLOCK_SIZE="${DSPARK_BLOCK_SIZE:-7}"

# The model ships its own chat_template.jinja in the checkpoint (fetched
# from the HF repo at image build time) -- no template mount needed,
# unlike the sibling qwen3.8 deploys, which mount a customized template.

# GB10's docker GPU access can be CDI-only on some DGX OS setups; the
# sibling qwen3.8 deploys use --gpus all successfully on this box, so that
# stays the default. If the container can't see the GPU, override, e.g.:
#   GPU_DOCKER_ARG="--device nvidia.com/gpu=all" ./start.sh
GPU_DOCKER_ARG="${GPU_DOCKER_ARG:---gpus all}"

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

mkdir -p "${HF_HOME}" "${CACHE_DIR}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "Log: ${LOG_FILE}"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

# NOTE: no "@${REV}" suffix -- this image's server args pass the string
# straight to transformers, which rejects '@' in repo ids (HFValidationError).
MODEL_PATH="${MODEL_ID}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_ID}"

SGLANG_ARGS=(
  -m sglang.launch_server
  --model-path "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --tp-size 1
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --cuda-graph-max-bs-decode "${CUDA_GRAPH_MAX_BS_DECODE}"
  --reasoning-parser qwen3
  --tool-call-parser minicpm5
  --host "${HOST}"
  --port "${PORT}"
)

if [[ "${ENABLE_DSPARK}" == "1" ]]; then
  SGLANG_ARGS+=(
    --trust-remote-code
    --speculative-algorithm DSPARK
    --speculative-draft-model-path "${DRAFT_MODEL_PATH}"
    --speculative-dspark-block-size "${DSPARK_BLOCK_SIZE}"
  )
fi

echo "Starting SGLang container for ${MODEL_ID} (native BF16)"
echo "Image: ${IMAGE}"
echo "Model path: ${MODEL_PATH}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Listening on ${HOST}:${PORT}"
echo "mem-fraction-static: ${MEM_FRACTION_STATIC}  cuda-graph-max-bs-decode: ${CUDA_GRAPH_MAX_BS_DECODE}"
echo "DSpark: ${ENABLE_DSPARK} $([[ "${ENABLE_DSPARK}" == "1" ]] && echo "(${DRAFT_MODEL_PATH}, block-size ${DSPARK_BLOCK_SIZE})")"
echo "Writing progress to ${LOG_FILE}"

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching SGLang container (MiniCPM5-2B, native BF16)
EOF

docker run -d \
  --name "${CONTAINER_NAME}" \
  ${GPU_DOCKER_ARG} \
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

echo "Waiting for HTTP readiness at ${READY_URL} (first start pulls image + ~5GB model)"
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
