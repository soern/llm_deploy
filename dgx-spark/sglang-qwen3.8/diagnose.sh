#!/usr/bin/env bash
# Diagnostics for the sglang-qwen3.8-dgx-spark deploy. Read-only: does not
# stop/remove any running container, does not touch caches, does not pull
# the image unless you pass --allow-pull (the image can be several GB).
#
# Usage (on the DGX Spark box, from this folder):
#   ./diagnose.sh                 > diagnose.log 2>&1   # then paste diagnose.log back
#   ./diagnose.sh --allow-pull    > diagnose.log 2>&1   # also lets throwaway
#                                                        # `docker run` checks
#                                                        # pull the image if
#                                                        # it isn't cached
#
# Everything is written to stdout so a single redirect captures the whole
# log; nothing here requires sudo.

set -uo pipefail

ALLOW_PULL=0
for arg in "$@"; do
  case "$arg" in
    --allow-pull) ALLOW_PULL=1 ;;
  esac
done

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-lmsysorg/sglang:dev-qwen38-27b-dflash2}"
CONTAINER_NAME="${CONTAINER_NAME:-sglang-qwen3.8-dspark}"

section() {
  echo
  echo "===== $1 ====="
}

run() {
  # run <description> -- <command...>
  # Always prints something, even on failure, and never aborts the script.
  local desc="$1"; shift
  echo "--- ${desc} ---"
  if "$@" 2>&1; then
    :
  else
    echo "(exit code: $?)"
  fi
}

echo "sglang-qwen3.8-dgx-spark diagnostics"
echo "generated: $(date -Is)"
echo "host: $(hostname 2>/dev/null || echo unknown)"

section "Host basics"
run "uname" uname -a
run "arch" arch
run "free -h" free -h
run "MemAvailable (unified-memory hw; nvidia-smi does not report memory here)" \
  bash -c "grep -E 'MemTotal|MemAvailable|SwapTotal' /proc/meminfo"
run "disk space (repo + docker root)" bash -c "df -h '${WORK_DIR}' /var/lib/docker 2>/dev/null || df -h '${WORK_DIR}'"
run "earlyoom recent kills (last 2h, no sudo)" \
  bash -c "journalctl -u earlyoom --no-pager --since '-2 hours' 2>&1 | tail -n 50"

section "GPU / CDI visibility"
if command -v nvidia-smi >/dev/null 2>&1; then
  run "nvidia-smi -L" nvidia-smi -L
  run "nvidia-smi (driver/CUDA version header only)" bash -c "nvidia-smi | head -n 15"
else
  echo "nvidia-smi not on PATH"
fi
if command -v nvidia-ctk >/dev/null 2>&1; then
  run "nvidia-ctk cdi list" nvidia-ctk cdi list
else
  echo "nvidia-ctk not on PATH"
fi
run "CDI spec files" bash -c "ls -la /etc/cdi /var/run/cdi 2>&1"

section "Docker basics"
run "docker version" docker version
run "docker info (runtimes)" bash -c "docker info --format '{{json .Runtimes}}' 2>&1"

section "Target image: ${IMAGE}"
if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "image is present in local cache"
  run "local image Created/Id/RepoDigests" \
    docker image inspect "${IMAGE}" --format '{{.Created}}  {{.Id}}  {{.RepoDigests}}'
  run "docker images (sglang rows)" bash -c "docker images | grep -i sglang || true"
  echo "--- docker manifest inspect (compares local tag to what the registry serves NOW; needs experimental/buildx manifest support) ---"
  docker manifest inspect "${IMAGE}" >/tmp/diagnose-manifest.$$ 2>&1
  mstatus=$?
  if [[ ${mstatus} -eq 0 ]]; then
    echo "remote manifest digest(s):"
    grep -i digest /tmp/diagnose-manifest.$$ | head -n 5
    echo "NOTE: compare this against the local RepoDigests above -- if they differ,"
    echo "the moving :dev-* tag has been rebuilt since you last pulled it."
  else
    echo "manifest inspect failed (exit ${mstatus}); output:"
    cat /tmp/diagnose-manifest.$$
  fi
  rm -f /tmp/diagnose-manifest.$$
  IMAGE_LOCAL=1
else
  echo "image is NOT present in local cache"
  IMAGE_LOCAL=0
  if [[ "${ALLOW_PULL}" -ne 1 ]]; then
    echo "(skipping the python-import checks below -- they'd trigger an implicit"
    echo " multi-GB pull. Re-run with --allow-pull if you want that.)"
  fi
fi

if [[ "${IMAGE_LOCAL}" -eq 1 || "${ALLOW_PULL}" -eq 1 ]]; then
  section "Isolated import checks inside the image (no GPU attached, no server boot)"
  echo "These re-create the exact failure from a fresh, throwaway container so we"
  echo "can see precisely what's broken without waiting through the ~7-9min boot."

  run "sglang package imports at all + its file location" \
    docker run --rm --entrypoint python3 "${IMAGE}" -c \
      "import sglang; print(sglang.__file__)"

  run "sglang version string (if exposed)" \
    docker run --rm --entrypoint python3 "${IMAGE}" -c \
      "import sglang; print(getattr(sglang, '__version__', 'no __version__ attr'))"

  run "pip-reported sglang/flashinfer/torch/transformers versions" \
    docker run --rm --entrypoint pip "${IMAGE}" freeze 2>&1 | grep -iE '^(sglang|flashinfer|torch|transformers)='

  run "git commit baked into the image (if the source tree has .git)" \
    docker run --rm --entrypoint sh "${IMAGE}" -c \
      'cd /sgl-workspace/sglang 2>/dev/null && git rev-parse HEAD 2>/dev/null || echo "no git info at /sgl-workspace/sglang"'

  run "names actually present in modelslim.schemes inside this image" \
    docker run --rm --entrypoint python3 "${IMAGE}" -c \
      "import sglang.srt.layers.quantization.modelslim.schemes as s; print(sorted(n for n in dir(s) if not n.startswith('_')))"

  run "names actually present in modelslim.schemes.__all__ (if defined)" \
    docker run --rm --entrypoint python3 "${IMAGE}" -c \
      "import sglang.srt.layers.quantization.modelslim.schemes as s; print(getattr(s, '__all__', 'no __all__'))"

  run "reproduce the exact failing import in isolation (full traceback)" \
    docker run --rm --entrypoint python3 "${IMAGE}" -c \
      "from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig"

  run "does the quantization package import cleanly at all (this is what launch_server hits first)" \
    docker run --rm --entrypoint python3 "${IMAGE}" -c \
      "from sglang.srt.layers.quantization import QUANTIZATION_METHODS; print('OK, methods:', sorted(QUANTIZATION_METHODS.keys()))"
fi

section "Existing container: ${CONTAINER_NAME}"
if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  run "docker inspect (state)" \
    docker inspect "${CONTAINER_NAME}" --format \
      'Status={{.State.Status}} ExitCode={{.State.ExitCode}} OOMKilled={{.State.OOMKilled}} StartedAt={{.State.StartedAt}} FinishedAt={{.State.FinishedAt}}'
  run "docker logs (last 200 lines)" docker logs --tail 200 "${CONTAINER_NAME}"
else
  echo "no container named ${CONTAINER_NAME} exists (running or exited)"
fi

section "Local log file"
if [[ -f "${WORK_DIR}/.sglang.log" ]]; then
  run "tail -n 200 .sglang.log" tail -n 200 "${WORK_DIR}/.sglang.log"
else
  echo "no ${WORK_DIR}/.sglang.log present"
fi

section "Current env overrides in effect (unset = start.sh default)"
for v in MODEL_ID MODEL_REV DRAFT_MODEL_ID DRAFT_REV SERVED_MODEL_NAME IMAGE \
         MEM_FRACTION_STATIC MAX_RUNNING_REQUESTS FLASHINFER_AUTOTUNE \
         MAMBA_SSM_DTYPE GPU_DOCKER_ARG PORT HOST; do
  if [[ -n "${!v:-}" ]]; then
    echo "${v}=${!v}"
  else
    echo "${v}=(unset, uses start.sh default)"
  fi
done
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN=(set, ${#HF_TOKEN} chars, not printing value)"
else
  echo "HF_TOKEN=(unset)"
fi

section "Registry reachability"
run "DNS resolve registry-1.docker.io" bash -c "getent hosts registry-1.docker.io || nslookup registry-1.docker.io"
run "HTTPS reachability to Docker Hub registry" \
  bash -c "curl -fsS -o /dev/null -w 'http_code=%{http_code}\n' --max-time 10 https://registry-1.docker.io/v2/"

echo
echo "===== done ====="
