#!/usr/bin/env bash
set -euo pipefail

# llama.cpp `llama-server` launch script for Qwen/Qwen3.5-4B (text only by
# default) with its built-in MTP head for speculative decoding, on an NVIDIA
# Jetson Orin Nano (8GB unified RAM/VRAM) running JetPack 7 (L4T R39, CUDA
# 13). Alternative to ../qwen3-4b-instruct-2507 (stronger
# at coding and tool use, MTP, far smaller KV cache). All Jetson deploys use
# the same llama.cpp image and port, so run only one at a time. See
# README.md for memory sizing and the MTP setup.
#
# The image is built locally from ./Dockerfile on first run (sm_87, CUDA 13)
# and shared with the other Jetson deploys. Qwen3.5 MTP needs a llama.cpp
# from May 2026 or later; an older shared image is detected below (rebuild
# with REBUILD=1). Serves an OpenAI-compatible /v1 API, so nixclaw uses its
# `vllm` backend against it.

# unsloth's "-MTP-GGUF" repo keeps the MTP layer (blk.32.nextn.*,
# nextn_predict_layers=1); their plain Qwen3.5-4B-GGUF files drop it.
MODEL_REPO="${MODEL_REPO:-unsloth/Qwen3.5-4B-MTP-GGUF}"
MODEL_FILE="${MODEL_FILE:-Qwen3.5-4B-Q4_K_M.gguf}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-4b}"

# Vision encoder (0.67GB). 0 = text only (default), 1 = load it. It only
# runs for requests that contain images, so it doesn't change text decode
# speed; off saves memory for context. MTP together with image input needs a
# llama.cpp from 2026-10-10 or later (PR #30257).
ENABLE_MMPROJ="${ENABLE_MMPROJ:-0}"
MMPROJ_REPO="${MMPROJ_REPO:-${MODEL_REPO}}"
MMPROJ_FILE="${MMPROJ_FILE:-mmproj-F16.gguf}"

# MTP draft tokens per step, from the model's own MTP layer (no separate
# drafter file). Each draft token re-reads the tied 248K-vocab lm_head
# (~0.5GB) plus the MTP layer, ~20-25% of a full step, so 2 is the estimated
# optimum; measure 0-3 on the board (README "MTP"). 0 = off.
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-2}"

# Qwen3.5 thinks before answering by default. 1 = on, 0 = answer directly
# (requests can still override via chat_template_kwargs.enable_thinking).
THINKING="${THINKING:-1}"

IMAGE="${IMAGE:-nixclaw/llama-server-jetson-orin:local}"
CUDA_VERSION="${CUDA_VERSION:-13.0.1}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-master}"
BUILD_JOBS="${BUILD_JOBS:-3}"
REBUILD="${REBUILD:-0}"
CONTAINER_NAME="${CONTAINER_NAME:-llama-qwen3.5-4b-jetson}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
# Only 8 of the 32 layers use full attention (4 KV heads x 256); the other
# 24 are Gated DeltaNet with a fixed-size state. KV is ~17KB/token at q8_0
# (+~2KB for the MTP layer), so 98304 tokens is ~1.9GB.
CTX_SIZE="${CTX_SIZE:-98304}"
CACHE_TYPE="${CACHE_TYPE:-q8_0}"
# llama-server splits CTX_SIZE evenly across slots; 1 slot keeps the whole
# window for one request (extra requests queue server-side).
PARALLEL="${PARALLEL:-1}"
# Prompt-processing compute buffers are sized off batch/ubatch at load time;
# on this 8GB board that reservation, not steady-state KV, is what OOM-kills
# under a real prefill (see the LFM2.5 deploy). Same reduced defaults.
BATCH_SIZE="${BATCH_SIZE:-1024}"
UBATCH_SIZE="${UBATCH_SIZE:-256}"
# Qwen3.5 model card sampling (thinking: general tasks; non-thinking: the
# instruct values). Server defaults only; a request's own params override.
if [[ "${THINKING}" == "1" ]]; then
  TEMP="${TEMP:-1.0}"
  TOP_P="${TOP_P:-0.95}"
  PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.5}"
else
  TEMP="${TEMP:-0.7}"
  TOP_P="${TOP_P:-0.8}"
  PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.5}"
fi
TOP_K="${TOP_K:-20}"
# The NVIDIA container runtime auto-detects "csv" mode on this board (L4T),
# and in csv mode `--gpus all` is rejected. Use the runtime explicitly; the
# nvidia/cuda base image already sets NVIDIA_VISIBLE_DEVICES=all.
GPU_DOCKER_ARG="${GPU_DOCKER_ARG:---runtime nvidia}"

PID_FILE=".llama.pid"
LOG_FILE=".llama.log"
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${WORK_DIR}/.models"
READY_URL="http://127.0.0.1:${PORT}/health"

cd "${WORK_DIR}"

case "${ENABLE_MMPROJ}" in
  0|1) ;;
  *)
    echo "ENABLE_MMPROJ must be 0 or 1, got '${ENABLE_MMPROJ}'"
    exit 1
    ;;
esac

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

command -v docker >/dev/null 2>&1 || {
  echo "docker is not on PATH"
  exit 1
}

command -v curl >/dev/null 2>&1 || {
  echo "curl is not on PATH"
  exit 1
}

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "Log: ${LOG_FILE}"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

# The other Jetson deploys use the same port; with --network host the second
# server would fail to bind, and two models don't fit in 8GB anyway.
if curl -fsS "${READY_URL}" >/dev/null 2>&1; then
  echo "Something is already serving on port ${PORT} (another Jetson deploy?)."
  echo "Stop it first, e.g. ../qwen3-4b-instruct-2507/stop.sh"
  exit 1
fi

if [[ "${REBUILD}" == "1" ]] || ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  # Pin a branch/tag to its current commit: the Dockerfile's clone step only
  # re-runs when its command line changes, so building "master" again would
  # silently reuse the cached (old) clone.
  build_ref="${LLAMA_CPP_REF}"
  if ! [[ "${build_ref}" =~ ^[0-9a-f]{40}$ ]]; then
    build_ref="$(curl -fsS "https://api.github.com/repos/ggml-org/llama.cpp/commits/${LLAMA_CPP_REF}" \
      -H 'Accept: application/vnd.github.sha' 2>/dev/null || true)"
    if ! [[ "${build_ref}" =~ ^[0-9a-f]{40}$ ]]; then
      echo "Could not resolve llama.cpp ref '${LLAMA_CPP_REF}' to a commit (GitHub API)."
      echo "Set LLAMA_CPP_REF=<40-char commit sha> and try again."
      exit 1
    fi
  fi
  echo "Building ${IMAGE} (llama.cpp ${LLAMA_CPP_REF} = ${build_ref}, CUDA ${CUDA_VERSION}, sm_87)"
  echo "This compiles CUDA kernels on the Orin -- expect 20-60 minutes on first build."
  docker build \
    --build-arg "CUDA_VERSION=${CUDA_VERSION}" \
    --build-arg "LLAMA_CPP_REF=${build_ref}" \
    --build-arg "BUILD_JOBS=${BUILD_JOBS}" \
    -t "${IMAGE}" \
    "${WORK_DIR}"
fi

# The image is shared with the other Jetson deploys and may predate llama.cpp's
# Qwen3.5 MTP support (May 2026). llama-server links libcuda, which only the
# NVIDIA runtime provides, so even --help needs GPU_DOCKER_ARG.
if (( NUM_SPECULATIVE_TOKENS > 0 )) \
    && ! docker run --rm ${GPU_DOCKER_ARG} "${IMAGE}" --help 2>&1 | grep -q 'draft-mtp'; then
  echo "${IMAGE} (llama.cpp $(docker run --rm --entrypoint cat "${IMAGE}" /LLAMA_CPP_COMMIT)) has no MTP support (--spec-type draft-mtp)."
  echo "Rebuild it (affects the other Jetson deploys too): REBUILD=1 ./start.sh"
  echo "Or start without MTP: NUM_SPECULATIVE_TOKENS=0 ./start.sh"
  exit 1
fi

mkdir -p "${MODELS_DIR}"

# download REPO FILE: fetch into .models/ under the file's base name
# (resumable); prints the local name.
download() {
  local repo="$1" file="$2"
  local name
  name="$(basename "${file}")"
  if [[ ! -f "${MODELS_DIR}/${name}" ]]; then
    local url="https://huggingface.co/${repo}/resolve/main/${file}"
    echo "Downloading ${url}" >&2
    local auth_args=()
    if [[ -n "${HF_TOKEN:-}" ]]; then
      auth_args=(-H "Authorization: Bearer ${HF_TOKEN}")
    fi
    curl -fL --retry 5 -C - "${auth_args[@]}" \
      -o "${MODELS_DIR}/${name}.part" "${url}" >&2
    mv "${MODELS_DIR}/${name}.part" "${MODELS_DIR}/${name}"
  fi
  echo "${name}"
}

model_name="$(download "${MODEL_REPO}" "${MODEL_FILE}")"

LLAMA_ARGS=(
  --model "/models/${model_name}"
  --alias "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --n-gpu-layers 999
  --ctx-size "${CTX_SIZE}"
  --parallel "${PARALLEL}"
  --batch-size "${BATCH_SIZE}"
  --ubatch-size "${UBATCH_SIZE}"
  # Quantized V cache requires flash attention.
  --flash-attn on
  --cache-type-k "${CACHE_TYPE}"
  --cache-type-v "${CACHE_TYPE}"
  --temp "${TEMP}"
  --top-p "${TOP_P}"
  --top-k "${TOP_K}"
  --presence-penalty "${PRESENCE_PENALTY}"
  --reasoning "$([[ "${THINKING}" == "1" ]] && echo on || echo off)"
  # Use the GGUF's own chat template, required for OpenAI tool calls.
  --jinja
)

if [[ "${ENABLE_MMPROJ}" == "1" ]]; then
  mmproj_name="$(download "${MMPROJ_REPO}" "${MMPROJ_FILE}")"
  LLAMA_ARGS+=(--mmproj "/models/${mmproj_name}")
fi

if (( NUM_SPECULATIVE_TOKENS > 0 )); then
  # No --spec-draft-model: the MTP context runs on the main model's weights.
  LLAMA_ARGS+=(
    --spec-type draft-mtp
    --spec-draft-n-max "${NUM_SPECULATIVE_TOKENS}"
  )
fi

echo "Starting llama-server container (Jetson Orin Nano, CUDA) for ${model_name}"
echo "Image: ${IMAGE} (llama.cpp $(docker run --rm --entrypoint cat "${IMAGE}" /LLAMA_CPP_COMMIT))"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Context: ${CTX_SIZE} tokens, KV cache ${CACHE_TYPE}, ${PARALLEL} slot(s), batch ${BATCH_SIZE}/${UBATCH_SIZE}"
echo "Vision: $([[ "${ENABLE_MMPROJ}" == "1" ]] && echo on || echo off)  MTP draft tokens: ${NUM_SPECULATIVE_TOKENS}  Thinking: $([[ "${THINKING}" == "1" ]] && echo on || echo off)"
echo "Listening on ${HOST}:${PORT}"
echo "Writing progress to ${LOG_FILE}"

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching llama-server container (Jetson Orin Nano, ${model_name}, mmproj=${ENABLE_MMPROJ}, mtp=${NUM_SPECULATIVE_TOKENS}, thinking=${THINKING})
EOF

docker run -d \
  --name "${CONTAINER_NAME}" \
  ${GPU_DOCKER_ARG} \
  --network host \
  -v "${MODELS_DIR}:/models:ro" \
  "${IMAGE}" \
  "${LLAMA_ARGS[@]}" \
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

echo "Waiting for model load at ${READY_URL}"
until curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "llama-server container exited before becoming ready"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  echo "  still starting..."
  sleep 2
done

# Fail loudly if it silently fell back to CPU.
if docker logs "${CONTAINER_NAME}" 2>&1 | grep -qE 'offloaded [1-9][0-9]*/[0-9]+ layers to GPU'; then
  docker logs "${CONTAINER_NAME}" 2>&1 | grep -E 'CUDA devices|Device 0|offloaded .* layers to GPU' || true
else
  echo "WARNING: no layers were offloaded to the GPU -- inference is running on CPU."
  echo "Check 'docker logs ${CONTAINER_NAME} | grep -i cuda' and README.md Troubleshooting."
fi

echo "llama-server is ready"
echo "OpenAI base URL: http://${HOST}:${PORT}/v1"
echo "Set config/agent.yaml: backend: \"vllm\", vllm.host: \"http://<jetson>:${PORT}\", vllm.model: \"${SERVED_MODEL_NAME}\", vllm.options.context_window: ${CTX_SIZE}"
