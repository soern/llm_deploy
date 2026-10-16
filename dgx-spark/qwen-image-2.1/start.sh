#!/usr/bin/env bash
set -euo pipefail

# Qwen-Image-2.1 image-generation server launch script for the DGX Spark
# (GB10, ARM64, sm_121a) box that already hosts vLLM. Builds the image from
# this directory's Dockerfile (nvcr.io/nvidia/pytorch base - no public image
# exists for this stack) and runs it alongside vLLM, sharing the GPU.
#
# Structured like deploy/dgx-spark/flux2-klein/start.sh: named container,
# host networking, host-side cache bind mounts, PID/log files next to this
# script, readiness poll with an exited-container check.
#
# Drop-in replacement for the FLUX.2-klein service on the same host port
# (default 8002). Serves Qwen-Image-2.1-Turbo (8 fixed steps) by default;
# IMAGE_MODEL_ID=Qwen/Qwen-Image-2.1 NUM_INFERENCE_STEPS=40 for the base
# model. DiT/VAE in bf16; text encoder FP8-stored by default
# (TEXT_ENCODER_QUANT=none restores the bf16 encoder).
#
# NO_CACHE=1 rebuilds without Docker's layer cache - needed once when the
# cached diffusers install predates Turbo support (the build then fails its
# sample_sigmas check).

IMAGE="qwen-image-2.1:local"
CONTAINER_NAME="qwen-image-2.1"
HOST="0.0.0.0"
PORT="${QWEN_IMAGE_PORT:-8002}"
PID_FILE=".qwen-image.pid"
LOG_FILE=".qwen-image.log"
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_HOME="${WORK_DIR}/.cache/huggingface"
READY_URL="http://127.0.0.1:${PORT}/health"

IMAGE_MODEL_ID="${IMAGE_MODEL_ID:-Qwen/Qwen-Image-2.1-Turbo}"
# Ignored by Turbo (its checkpoint fixes 8 steps); the base model wants 40.
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-8}"
NO_CACHE="${NO_CACHE:-0}"
TRUE_CFG_SCALE="${TRUE_CFG_SCALE:-1.0}"
# fp8 (default): unsloth/Qwen-Image-2.1-FP8 text encoder, ~8 GB less resident.
# none: bf16 text encoder from IMAGE_MODEL_ID.
TEXT_ENCODER_QUANT="${TEXT_ENCODER_QUANT:-fp8}"
MAX_EDIT_LONG_EDGE="${MAX_EDIT_LONG_EDGE:-2048}"
# cuDNN SDPA measured ~1.06x over the default backend on this model's actual
# (unmasked) attention path at seq=16384 on GB10 - small but free.
ATTENTION_BACKEND="${ATTENTION_BACKEND:-_native_cudnn}"
# Size-aware admission (see README): a request needs
#   n x (output_MP x DENOISE_MB_PER_MP + ref_MP x REF_MB_PER_MP)
#     + TILED_VAE_MB + ADMISSION_MARGIN_MB
# free in min(CUDA free, host MemAvailable); otherwise it waits up to
# ADMISSION_WAIT_SECONDS, then 503 naming the largest size that fits.
# 1024x1024 needs ~3.3 GB, 2048x2048 ~4.0 GB, a 1 MP edit ~5.8 GB.
# Values measured with the allocator on 2026-10-11.
DENOISE_MB_PER_MP="${DENOISE_MB_PER_MP:-250}"
REF_MB_PER_MP="${REF_MB_PER_MP:-2400}"
TILED_VAE_MB="${TILED_VAE_MB:-1000}"
ADMISSION_MARGIN_MB="${ADMISSION_MARGIN_MB:-2000}"
ADMISSION_WAIT_SECONDS="${ADMISSION_WAIT_SECONDS:-60}"
# VAE tiling: auto = untiled (no seams) only when the pool has room right
# before each decode/encode, else tiled. Untiled decode measured 7.2 GB/MP
# (~30 GB at 4.2 MP; froze the box 2026-10-08 and 2026-10-10), untiled encode
# 2.4 GB/MP, tiled ~0.5 GB. 1 = always tiled, 0 = never (A/B testing only).
VAE_TILING="${VAE_TILING:-auto}"
UNTILED_VAE_MB_PER_MP="${UNTILED_VAE_MB_PER_MP:-7500}"
UNTILED_VAE_ENCODE_MB_PER_MP="${UNTILED_VAE_ENCODE_MB_PER_MP:-2600}"
UNTILED_VAE_MARGIN_MB="${UNTILED_VAE_MARGIN_MB:-2000}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# OOM kill order on the Spark: qwen3-tts (+800) first, then this container
# (+500), sglang (-900) last - rather than dbus/systemd or the LLM.
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"
# Restart after a crash, but at most twice: an OOM-killed server that comes
# straight back reloads ~23 GB into a box that is already short of memory.
# Not restarted after a reboot or `docker stop` - start it again by hand.
RESTART_POLICY="${RESTART_POLICY:-on-failure:2}"

command -v docker >/dev/null 2>&1 || {
  echo "docker is not on PATH"
  exit 1
}

command -v curl >/dev/null 2>&1 || {
  echo "curl is not on PATH"
  exit 1
}

mkdir -p "${HF_HOME}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} is already running"
    echo "Log: ${LOG_FILE}"
    exit 0
  fi
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

echo "Building ${IMAGE} from ${WORK_DIR}"
build_args=()
if [[ "${NO_CACHE}" == "1" ]]; then
  build_args+=(--no-cache)
fi
docker build "${build_args[@]}" -t "${IMAGE}" "${WORK_DIR}"

echo "Starting Qwen-Image-2.1 container for ${IMAGE_MODEL_ID}"
echo "DiT bf16, text encoder: ${TEXT_ENCODER_QUANT}, steps: ${NUM_INFERENCE_STEPS}, true_cfg_scale: ${TRUE_CFG_SCALE}"
echo "Attention backend: ${ATTENTION_BACKEND}, admission: ${DENOISE_MB_PER_MP} MB/MP output + ${REF_MB_PER_MP} MB/MP refs + ${TILED_VAE_MB} + ${ADMISSION_MARGIN_MB} MB (wait ${ADMISSION_WAIT_SECONDS}s), vae tiling: ${VAE_TILING} (untiled margin ${UNTILED_VAE_MARGIN_MB} MB), oom_score_adj: ${OOM_SCORE_ADJ}"
echo "Listening on ${HOST}:${PORT}"
echo "Writing progress to ${LOG_FILE}"

cat >"${LOG_FILE}" <<EOF
[$(date -Is)] launching qwen-image-2.1 container
EOF

docker run -d \
  --name "${CONTAINER_NAME}" \
  --network host \
  --ipc host \
  --gpus all \
  --oom-score-adj "${OOM_SCORE_ADJ}" \
  --restart "${RESTART_POLICY}" \
  -e IMAGE_MODEL_ID="${IMAGE_MODEL_ID}" \
  -e NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
  -e TRUE_CFG_SCALE="${TRUE_CFG_SCALE}" \
  -e TEXT_ENCODER_QUANT="${TEXT_ENCODER_QUANT}" \
  -e MAX_EDIT_LONG_EDGE="${MAX_EDIT_LONG_EDGE}" \
  -e ATTENTION_BACKEND="${ATTENTION_BACKEND}" \
  -e DENOISE_MB_PER_MP="${DENOISE_MB_PER_MP}" \
  -e REF_MB_PER_MP="${REF_MB_PER_MP}" \
  -e TILED_VAE_MB="${TILED_VAE_MB}" \
  -e ADMISSION_MARGIN_MB="${ADMISSION_MARGIN_MB}" \
  -e ADMISSION_WAIT_SECONDS="${ADMISSION_WAIT_SECONDS}" \
  -e VAE_TILING="${VAE_TILING}" \
  -e UNTILED_VAE_MB_PER_MP="${UNTILED_VAE_MB_PER_MP}" \
  -e UNTILED_VAE_ENCODE_MB_PER_MP="${UNTILED_VAE_ENCODE_MB_PER_MP}" \
  -e UNTILED_VAE_MARGIN_MB="${UNTILED_VAE_MARGIN_MB}" \
  -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
  -e PORT="${PORT}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HF_HOME=/root/.cache/huggingface \
  -v "${HF_HOME}:/root/.cache/huggingface" \
  "${IMAGE}" \
  >/dev/null

container_id="$(docker inspect -f '{{.Id}}' "${CONTAINER_NAME}")"
echo "${container_id}" > "${PID_FILE}"
echo "Spawned container ${CONTAINER_NAME} (${container_id})"

log_follow_pid=""
cleanup() {
  if [[ -n "${log_follow_pid}" ]] && kill -0 "${log_follow_pid}" 2>/dev/null; then
    kill "${log_follow_pid}"
  fi
}
trap cleanup EXIT

(docker logs -f "${CONTAINER_NAME}" >> "${LOG_FILE}" 2>&1) &
log_follow_pid=$!

echo "Waiting for HTTP readiness at ${READY_URL}"
echo "(first run downloads ~16GB of DiT/VAE weights + the 9.4GB FP8 text encoder - this can take a while)"
until curl -fsS "${READY_URL}" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    echo "qwen-image-2.1 container exited before becoming ready"
    tail -n 200 "${LOG_FILE}" || true
    exit 1
  fi
  echo "  still starting..."
  sleep 10
done

echo "qwen-image-2.1 is ready"
echo "OpenAI-shaped base URL: http://${HOST}:${PORT}/v1"
