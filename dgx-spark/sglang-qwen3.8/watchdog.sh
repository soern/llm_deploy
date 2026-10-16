#!/usr/bin/env bash
# Watchdog for the SGLang container: brings the LLM back after a crash, an
# OOM kill or a hang, and keeps memory free for it - this is the service on
# the Spark that matters most. Runs on the host (see sglang-watchdog.service).
#
# What it does, every CHECK_INTERVAL seconds:
#   - container exited (crash, OOM kill, reboot): start it again, once enough
#     memory is free (see "making room" below).
#   - container running, /health failing for FAIL_THRESHOLD checks in a row
#     AND no "Prefill batch"/"Decode batch" log line for STALL_SECONDS: it is
#     hung (2026-10-09: HTTP server up, scheduler silent for 3 h) -> stop and
#     start it. Both conditions are needed, because /health also fails while
#     the server is merely saturated (it queues a tiny generation), and then
#     the batch log lines keep coming.
#   - while starting (no successful /health since the container started), it
#     waits up to STARTUP_TIMEOUT (boot incl. torch.compile and CUDA graph
#     capture takes ~10-15 min) before treating it as hung.
#   - host MemAvailable below PROTECT_MIN_AVAILABLE_MB on two checks in a row
#     while SGLang runs: stop the YIELD_CONTAINERS (image generation, TTS) so
#     the box doesn't swap itself into a stall like 2026-10-09.
#
# Making room: before starting SGLang it waits up to MEMORY_WAIT_SECONDS for
# REQUIRED_AVAILABLE_MB of MemAvailable (memory of a killed process is freed
# with a delay), then stops YIELD_CONTAINERS one by one until there is room.
# If there still isn't, it does not start SGLang (it would only thrash) and
# tries again on the next check.
#
# Containers it stopped are started again once SGLang is healthy and
# MemAvailable >= RESTORE_MIN_AVAILABLE_MB, at the earliest RESTORE_DELAY
# seconds after they were stopped, one per check.
#
# At most MAX_RECOVERIES_PER_HOUR restarts per hour; after that it only logs,
# so a model that can't come up doesn't loop forever.
#
# To stop SGLang on purpose, use ./stop.sh (it removes the container, which
# the watchdog leaves alone) or stop the watchdog first. A plain
# `docker stop` looks like a crash and the watchdog starts it again.
#
# NOTIFY_CMD (optional) runs with the message as its only argument, e.g. a
# script that sends a Signal message.

set -uo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-sglang-qwen3.8-dspark}"
PORT="${PORT:-8000}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:${PORT}/health}"
CHECK_INTERVAL="${CHECK_INTERVAL:-30}"
# SGLang's /health waits up to 20 s for its test generation.
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-30}"
FAIL_THRESHOLD="${FAIL_THRESHOLD:-10}"
STALL_SECONDS="${STALL_SECONDS:-300}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
STOP_TIMEOUT="${STOP_TIMEOUT:-60}"
# SGLang with MAX_TOTAL_TOKENS=262144 / MAX_MAMBA_CACHE_SIZE=32 measured
# ~56 GB after its first request (2026-10-09), plus margin.
REQUIRED_AVAILABLE_MB="${REQUIRED_AVAILABLE_MB:-62000}"
MEMORY_WAIT_SECONDS="${MEMORY_WAIT_SECONDS:-120}"
# Lower-priority containers, stopped first when SGLang needs room (in this
# order; empty = never stop anything).
YIELD_CONTAINERS="${YIELD_CONTAINERS-qwen3-tts qwen-image-2.1}"
PROTECT_MIN_AVAILABLE_MB="${PROTECT_MIN_AVAILABLE_MB:-3000}"
# Measured 2026-10-09: SGLang + qwen-image loaded leave ~28 GB; TTS needs
# ~10 GB incl. its peak. 20 GB lets TTS come back next to qwen-image and
# still leaves room for an image generation.
RESTORE_MIN_AVAILABLE_MB="${RESTORE_MIN_AVAILABLE_MB:-20000}"
RESTORE_DELAY="${RESTORE_DELAY:-1800}"
MAX_RECOVERIES_PER_HOUR="${MAX_RECOVERIES_PER_HOUR:-3}"
NOTIFY_CMD="${NOTIFY_CMD-}"

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# name -> epoch seconds when the watchdog stopped it
STATE_FILE="${STATE_FILE:-${WORK_DIR}/.watchdog-yielded}"

fails=0
low_memory_checks=0
healthy_since_start=0
container_seen_missing=0
rate_limited_logged=0
recoveries=()

log() {
  echo "$(date -Is) $*"
}

notify() {
  log "$*"
  if [[ -n "${NOTIFY_CMD}" ]]; then
    ${NOTIFY_CMD} "sglang-watchdog: $*" >/dev/null 2>&1 &
  fi
}

available_mb() {
  awk '/^MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo
}

container_field() {
  docker inspect -f "$1" "$2" 2>/dev/null
}

is_running() {
  [[ "$(container_field '{{.State.Running}}' "$1")" == "true" ]]
}

healthy() {
  curl -fsS -m "${HEALTH_TIMEOUT}" -o /dev/null "${HEALTH_URL}" 2>/dev/null
}

making_progress() {
  # Any batch log line in the last STALL_SECONDS means the scheduler works.
  docker logs --since "${STALL_SECONDS}s" "${CONTAINER_NAME}" 2>&1 \
    | grep -qE 'Prefill batch|Decode batch'
}

seconds_since_start() {
  local started
  started="$(container_field '{{.State.StartedAt}}' "${CONTAINER_NAME}")" || return 1
  echo $(( $(date +%s) - $(date -d "${started}" +%s) ))
}

record_yield() {
  touch "${STATE_FILE}"
  grep -v "^$1 " "${STATE_FILE}" > "${STATE_FILE}.tmp" || true
  echo "$1 $(date +%s)" >> "${STATE_FILE}.tmp"
  mv "${STATE_FILE}.tmp" "${STATE_FILE}"
}

forget_yield() {
  [[ -f "${STATE_FILE}" ]] || return 0
  grep -v "^$1 " "${STATE_FILE}" > "${STATE_FILE}.tmp" || true
  mv "${STATE_FILE}.tmp" "${STATE_FILE}"
}

# Stop lower-priority containers until MemAvailable >= $1 MB. Returns 0 if
# that was reached.
yield_until() {
  local need="$1" c
  for c in ${YIELD_CONTAINERS}; do
    (( $(available_mb) >= need )) && return 0
    if is_running "${c}"; then
      notify "stopping ${c} to free memory for SGLang (MemAvailable $(available_mb) MB, need ${need} MB)"
      docker stop -t 30 "${c}" >/dev/null 2>&1
      record_yield "${c}"
      sleep 10  # unified memory is handed back with a delay
    fi
  done
  (( $(available_mb) >= need ))
}

wait_for_memory() {
  local deadline=$(( $(date +%s) + MEMORY_WAIT_SECONDS ))
  while (( $(available_mb) < REQUIRED_AVAILABLE_MB )) && (( $(date +%s) < deadline )); do
    sleep 5
  done
  (( $(available_mb) >= REQUIRED_AVAILABLE_MB )) || yield_until "${REQUIRED_AVAILABLE_MB}"
}

recovery_allowed() {
  local now kept=() t
  now="$(date +%s)"
  for t in "${recoveries[@]+"${recoveries[@]}"}"; do
    (( now - t < 3600 )) && kept+=("${t}")
  done
  recoveries=("${kept[@]+"${kept[@]}"}")
  (( ${#recoveries[@]} < MAX_RECOVERIES_PER_HOUR ))
}

recover() {
  local reason="$1"
  if ! recovery_allowed; then
    if (( ! rate_limited_logged )); then
      notify "would restart SGLang (${reason}), but ${MAX_RECOVERIES_PER_HOUR} restarts in the last hour already - waiting"
      rate_limited_logged=1
    fi
    return
  fi
  rate_limited_logged=0
  if is_running "${CONTAINER_NAME}"; then
    notify "SGLang ${reason} - stopping it"
    docker stop -t "${STOP_TIMEOUT}" "${CONTAINER_NAME}" >/dev/null 2>&1
  fi
  if ! wait_for_memory; then
    log "not starting SGLang (${reason}): MemAvailable $(available_mb) MB < ${REQUIRED_AVAILABLE_MB} MB even after stopping ${YIELD_CONTAINERS} - retrying next check"
    return
  fi
  recoveries+=("$(date +%s)")
  notify "starting SGLang (${reason}); MemAvailable $(available_mb) MB"
  docker start "${CONTAINER_NAME}" >/dev/null 2>&1 || log "docker start ${CONTAINER_NAME} failed"
  fails=0
  healthy_since_start=0
}

restore_yielded() {
  [[ -s "${STATE_FILE}" ]] || return 0
  local now name stopped_at
  now="$(date +%s)"
  while read -r name stopped_at; do
    [[ -n "${name}" ]] || continue
    if is_running "${name}"; then
      forget_yield "${name}"  # someone started it already
      continue
    fi
    (( now - stopped_at >= RESTORE_DELAY )) || continue
    (( $(available_mb) >= RESTORE_MIN_AVAILABLE_MB )) || return 0
    notify "starting ${name} again (stopped by the watchdog; MemAvailable $(available_mb) MB)"
    docker start "${name}" >/dev/null 2>&1 || log "docker start ${name} failed"
    forget_yield "${name}"
    return 0  # one per check, so its memory shows up before the next
  done < "${STATE_FILE}"
}

check() {
  local status
  status="$(container_field '{{.State.Status}}' "${CONTAINER_NAME}")" || status="missing"

  case "${status}" in
    missing)
      # Removed on purpose (stop.sh) or never created: not ours to start.
      if (( ! container_seen_missing )); then
        log "container ${CONTAINER_NAME} does not exist - start it with ./start.sh; idle until then"
        container_seen_missing=1
      fi
      return
      ;;
    running)
      ;;
    restarting|paused)
      return
      ;;
    *)
      container_seen_missing=0
      local code oom
      code="$(container_field '{{.State.ExitCode}}' "${CONTAINER_NAME}")"
      oom="$(container_field '{{.State.OOMKilled}}' "${CONTAINER_NAME}")"
      recover "exited (status ${status}, exit code ${code}, cgroup OOM ${oom})"
      return
      ;;
  esac
  container_seen_missing=0

  # Protect a running SGLang from the rest of the box.
  if (( $(available_mb) < PROTECT_MIN_AVAILABLE_MB )); then
    low_memory_checks=$(( low_memory_checks + 1 ))
    if (( low_memory_checks >= 2 )); then
      yield_until "${PROTECT_MIN_AVAILABLE_MB}" \
        || log "MemAvailable $(available_mb) MB still below ${PROTECT_MIN_AVAILABLE_MB} MB with ${YIELD_CONTAINERS} stopped"
      low_memory_checks=0
    fi
  else
    low_memory_checks=0
  fi

  if healthy; then
    if (( ! healthy_since_start )); then
      log "SGLang healthy"
      healthy_since_start=1
    fi
    fails=0
    restore_yielded
    return
  fi

  local uptime
  uptime="$(seconds_since_start)" || uptime=0
  if (( ! healthy_since_start )); then
    if (( uptime > STARTUP_TIMEOUT )); then
      recover "not healthy ${uptime}s after start (limit ${STARTUP_TIMEOUT}s)"
    fi
    return
  fi

  fails=$(( fails + 1 ))
  if (( fails < FAIL_THRESHOLD )); then
    return
  fi
  if making_progress; then
    log "/health failing ${fails}x, but the scheduler is still logging batches - busy, not hung"
    fails=0
    return
  fi
  recover "hung: /health failed ${fails}x in a row and no batch log line for ${STALL_SECONDS}s"
}

log "watching ${CONTAINER_NAME} (${HEALTH_URL}) every ${CHECK_INTERVAL}s; needs ${REQUIRED_AVAILABLE_MB} MB to start; yields: ${YIELD_CONTAINERS}"
# A container that is already up and healthy when the watchdog starts.
if is_running "${CONTAINER_NAME}" && healthy; then
  healthy_since_start=1
fi
while true; do
  check
  sleep "${CHECK_INTERVAL}"
done
