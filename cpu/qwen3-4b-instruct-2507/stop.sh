#!/usr/bin/env bash
set -euo pipefail

# Stop the CPU-only Ollama container for Qwen/Qwen3-4B-Instruct-2507.
# Usage: ./stop.sh

CONTAINER_NAME="${CONTAINER_NAME:-ollama-qwen3-4b-cpu}"

if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Stopping ${CONTAINER_NAME}..."
  docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  echo "Container ${CONTAINER_NAME} stopped."
else
  echo "Container ${CONTAINER_NAME} is not running."
fi
