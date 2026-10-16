#!/bin/bash
set -e

CONTAINER_NAME="nixclaw-kokoro-tts-fr"

echo "Stopping Kokoro French TTS server..."
cd "$(dirname "${BASH_SOURCE[0]}")"
docker compose down
echo "Done."
