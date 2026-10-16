#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONTAINER_NAME="nixclaw-kokoro-tts-fr"
LOG_FILE="$SCRIPT_DIR/.kokoro-tts-fr.log"
HEALTH_URL="http://127.0.0.1:${TTS_PORT:-8008}/health"
TIMEOUT=300
POLL_INTERVAL=2

echo "[$(date)] Starting Kokoro French TTS server..."

# Check if container is already running
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}\$"; then
    echo "Container $CONTAINER_NAME is already running"
    exit 0
fi

# Remove any stopped container with the same name
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}\$"; then
    echo "Removing stopped container $CONTAINER_NAME..."
    docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

# Build and start
cd "$SCRIPT_DIR"
echo "[$(date)] Building Docker image..."
docker compose build kokoro-tts-fr

echo "[$(date)] Starting container..."
docker compose up -d kokoro-tts-fr

# Stream logs in background
docker logs -f "$CONTAINER_NAME" 2>&1 | tee "$LOG_FILE" &
LOGS_PID=$!

# Wait for health endpoint
echo "[$(date)] Waiting for server to be ready (polling $HEALTH_URL)..."
START_TIME=$(date +%s)

while true; do
    if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
        echo "[$(date)] Server is ready!"
        kill $LOGS_PID 2>/dev/null || true
        exit 0
    fi
    
    ELAPSED=$(($(date +%s) - START_TIME))
    if [ $ELAPSED -gt $TIMEOUT ]; then
        echo "[$(date)] Timeout waiting for server (${TIMEOUT}s exceeded)"
        echo "[$(date)] Last 50 lines of logs:"
        tail -50 "$LOG_FILE"
        kill $LOGS_PID 2>/dev/null || true
        exit 1
    fi
    
    echo "[$(date)] Still waiting... (${ELAPSED}s elapsed)"
    sleep $POLL_INTERVAL
done
