#!/usr/bin/env bash
set -euo pipefail

# Stop the Jetson Orin Nano llama-server container for google/gemma-4-E2B-it.
# Usage: ./stop.sh

CONTAINER_NAME="${CONTAINER_NAME:-llama-gemma4-e2b-jetson}"

if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Stopping ${CONTAINER_NAME}..."
  docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  echo "Container ${CONTAINER_NAME} stopped."
else
  echo "Container ${CONTAINER_NAME} is not running."
fi
