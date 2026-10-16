#!/usr/bin/env bash
set -euo pipefail

# SGLang launch script for LiquidAI/LFM2.5-2.6B on NVIDIA DGX Spark (GB10,
# sm_121a). See README.md for the full rationale, sourcing, and trust notes.
#
# The model is a DENSE hybrid architecture (Lfm2ForCausalLM): 32 layers,
# mostly gated causal-conv blocks plus full-attention layers, 131k context,
# native BF16, ~5.4GB (NOT the 8B-A1B MoE -- that is a separate checkpoint,
# LiquidAI/LFM2.5-8B-A1B). SGLang ships native LFM2 support
# (python/sglang/srt/models/lfm2.py + lfm2_dspark.py) with a `lfm2` tool-call
# parser; the model's chat template is Qwen-style (`...` thinking),
# so --reasoning-parser qwen3 applies.
#
# No SGLang cookbook cell verifies LFM2.5-2.6B on GB10 (the cookbook covers
# h100/h200/b200, and its variant list omits 2.6B). The flags below are the
# union of:
#   * the verified LFM2.5 dense-text cookbook cells (docs.sglang.io/cookbook/
#     autoregressive/LiquidAI/LFM2.5; source docs/src/snippets/configs/
#     LiquidAI/lfm2.5.jsx): --trust-remote-code, --tp 1, --reasoning-parser
#     qwen3, --tool-call-parser lfm2, dev-cu13 image
#   * the machine-validated ../sglang-qwen3.8 recipe on this exact
#     box (hybrid conv/mamba model with cgroup-capped unified memory):
#     --memory 100g cap, --mem-fraction-static 0.80, flashinfer attention
# Treat all of it as UNVERIFIED until a run on this box.
#
# DSpark speculative decoding is on by default: LiquidAI publishes a matching
# 655MB BF16 draft checkpoint (LiquidAI/LFM2.5-2.6B-DSpark, draft
# block_size 9, Qwen3-style 5-layer draft). Set ENABLE_DSPARK=0 to run the
# base model alone.
#
# Plain `docker run` against the official upstream image, same pattern as
# our sibling ../sglang-qwen3.8 and ../sglang-minicpm5-2b
# deploys -- no custom image build, no sudo.

MODEL_ID="${MODEL_ID:-LiquidAI/LFM2.5-2.6B}"
# Provenance only (same convention as ../sglang-qwen3.8): the revs
# below are NOT passed to --model-path. This image's server args hand the
# string straight to transformers, whose validate_repo_id rejects '@' -- an
# "repo@revision" path dies at startup with an HFValidationError.
MODEL_REV="${MODEL_REV:-654f9463ce32b05d0429d76fe1f580b27d4c1ac0}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-lfm2.5-2.6b}"
# Cookbook-mandated image for LFM2.5 (LFM2 support ships in the dev image;
# not yet in a tagged release). The sibling minicpm5 deploy uses plain :dev,
# but :dev-cu13 is the LFM2.5-validated line -- override IMAGE to switch.
IMAGE="${IMAGE:-lmsysorg/sglang:dev-cu13}"
CONTAINER_NAME="${CONTAINER_NAME:-sglang-lfm2.5-2.6b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# The only mem-fraction-static validated on this box is 0.80 (qwen3.8 hybrid
# model, same cgroup-capped unified-memory accounting). Do NOT raise toward
# 0.85+: the qwen3.8 README records earlyoom SIGTERMs at 0.85. Not verified
# for 2.6B -- if boot OOMs, lower it (the 2.6B model itself needs only ~6GB
# of the KV+weights budget).
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.80}"

# DSpark: the draft checkpoint's config declares block_size 9 (not the 7 used
# by the qwen3.8 draft) -- the two must match.
ENABLE_DSPARK="${ENABLE_DSPARK:-1}"
DRAFT_MODEL_ID="${DRAFT_MODEL_ID:-LiquidAI/LFM2.5-2.6B-DSpark}"
DRAFT_REV="${DRAFT_REV:-458cedab07d0f7b2b05700c77e1aa463d43d6f04}"
DSPARK_BLOCK_SIZE="${DSPARK_BLOCK_SIZE:-9}"

# The model ships its own chat_template.jinja in the checkpoint -- no
# template mount needed, same as the sibling minicpm5 deploy.

# GB10's docker GPU access can be CDI-only on some DGX OS setups; the sibling
# qwen3.8 deploys use --gpus all successfully on this box, so that stays the
# default. If the container can't see the GPU, override, e.g.:
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
  --trust-remote-code
  --model-path "${MODEL_PATH}"
  --tp 1
  --served-model-name "${SERVED_MODEL_NAME}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --attention-backend flashinfer
  --reasoning-parser qwen3
  --tool-call-parser lfm2
  --host "${HOST}"
  --port "${PORT}"
)

if [[ "${ENABLE_DSPARK}" == "1" ]]; then
  SGLANG_ARGS+=(
    --speculative-algorithm DSPARK
    --speculative-draft-model-path "${DRAFT_MODEL_PATH}"
    --speculative-dspark-block-size "${DSPARK_BLOCK_SIZE}"
    --speculative-draft-model-quantization unquant
  )
fi

echo "Starting SGLang container for ${MODEL_ID} (native BF16, dense hybrid)"
echo "Image: ${IMAGE}"
echo "Model path: ${MODEL_PATH}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Listening on ${HOST}:${PORT}"
echo "mem-fraction-static: ${MEM_FRACTION_STATIC}"
if [[ "${ENABLE_DSPARK}" == "1" ]]; then
  echo "DSpark: on (${DRAFT_MODEL_PATH}, block-size ${DSPARK_BLOCK_SIZE})"
else
  echo "DSpark: off"
fi
echo "Chat template: model's built-in (no mount)"
echo "Writing progress to ${LOG_FILE}"

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching SGLang container (LiquidAI/LFM2.5-2.6B, dense hybrid)
EOF

# SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN: the checkpoint declares a
# 131072 context, but SGLang derives a 128000 cap and otherwise refuses to
# boot with a ValueError. 131k is the model's native context, so allow it.
docker run -d \
  --name "${CONTAINER_NAME}" \
  ${GPU_DOCKER_ARG} \
  --memory 100g \
  --memory-swap 100g \
  --shm-size 16g \
  --network host \
  --ipc host \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
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

echo "Waiting for HTTP readiness at ${READY_URL} (first start pulls image + ~5.4GB model)"
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