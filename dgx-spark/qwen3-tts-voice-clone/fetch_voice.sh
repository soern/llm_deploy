#!/usr/bin/env bash
set -euo pipefail

# Download a cloned voice from the voice-clone service:
#   NAME.safetensors  the voice itself - drop it into the TTS server's voices dir
#   NAME.preview.wav  the preview sentence (if one was made)
#   NAME.ref.wav      the exact reference audio the voice was built from
#
#   ./fetch_voice.sh anna                                  # -> ./voices/
#   ./fetch_voice.sh -o ../qwen3-tts/voices anna # install locally
#   ./fetch_voice.sh -o spark:nixclaw2/deploy/dgx-spark/qwen3-tts/voices anna  # scp
#
# On the Spark itself nothing needs fetching: start.sh points the service at
# the TTS server's voices dir, so a new voice is usable right away.

usage() {
  cat <<EOF
usage: $(basename "$0") [-u URL] [-o DIR] NAME
  -u URL   voice-clone service (default: \$QWEN_CLONE_URL or http://127.0.0.1:8011)
  -o DIR   destination (default: ./voices); user@host:path is copied with scp
EOF
  exit 2
}

url="${QWEN_CLONE_URL:-http://127.0.0.1:8011}"
out_dir="./voices"
while getopts "u:o:h" opt; do
  case "$opt" in
    u) url="$OPTARG" ;;
    o) out_dir="$OPTARG" ;;
    *) usage ;;
  esac
done
shift $((OPTIND - 1))
[[ $# -eq 1 ]] || usage
name="$1"

remote=""
if [[ "$out_dir" == *:* ]]; then
  remote="$out_dir"
  out_dir="$(mktemp -d)"
  trap 'rm -rf "$out_dir"' EXIT
fi
mkdir -p "$out_dir"

fetch() {  # fetch ENDPOINT FILE REQUIRED
  local status
  status="$(curl -sS -o "${out_dir}/$2.part" -w '%{http_code}' "${url%/}/v1/voices/${name}/$1")"
  if [[ "$status" == "200" ]]; then
    mv "${out_dir}/$2.part" "${out_dir}/$2"
    echo "  ${out_dir}/$2"
  else
    rm -f "${out_dir}/$2.part"
    if [[ "$3" == "required" ]]; then
      echo "could not fetch $2 (HTTP ${status})" >&2
      exit 1
    fi
  fi
}

echo "Fetching voice '${name}' from ${url}:"
fetch file "${name}.safetensors" required
fetch preview "${name}.preview.wav" optional
fetch reference "${name}.ref.wav" optional

if [[ -n "$remote" ]]; then
  echo "Copying to ${remote}"
  scp -q "$out_dir"/"${name}".* "$remote"/
fi
