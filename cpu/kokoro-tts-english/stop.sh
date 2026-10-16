#!/bin/bash
set -e

CONTAINER_NAME="nixclaw-kokoro-tts-en"

echo "Stopping Kokoro English TTS server..."
docker compose down
echo "Done."
