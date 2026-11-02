#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="moss-image"
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${WORK_DIR}/.moss-image.pid"
LOG_FILE="${WORK_DIR}/.moss-image.log"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Stopping MOSS-Image container"
  docker stop "${CONTAINER_NAME}" 2>/dev/null || true
  docker rm "${CONTAINER_NAME}"
else
  echo "Container ${CONTAINER_NAME} not found"
fi

rm -f "${PID_FILE}"
echo "Stopped. Log preserved at ${LOG_FILE}"
