#!/usr/bin/env bash
set -euo pipefail

# Stop the Jetson Orin Nano llama-server container for LiquidAI/LFM2.5-8B-A1B.
# Usage: ./stop.sh

CONTAINER_NAME="${CONTAINER_NAME:-llama-lfm2.5-8b-a1b-jetson}"

if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Stopping ${CONTAINER_NAME}..."
  docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  echo "Container ${CONTAINER_NAME} stopped."
else
  echo "Container ${CONTAINER_NAME} is not running."
fi
