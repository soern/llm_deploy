#!/usr/bin/env bash
set -euo pipefail

# Upload a reference clip to the voice-clone service, print what it made of it
# (transcript, reference length, timings), then download the cloned voice +
# preview + exact reference audio via fetch_voice.sh.
#
#   ./clone_voice.sh -l de ~/clips/anna.m4a anna
#   QWEN_CLONE_URL=http://gx10-811e.lan:8011 ./clone_voice.sh -l fr -o ./my-voices clip.wav marie
#
# Without -t the service transcribes the clip itself (faster-whisper) and trims
# the reference to <= -s seconds at sentence/word boundaries. With -t the clip
# is used as-is and must be <= 30s, and the transcript must match it exactly.

usage() {
  cat <<EOF
usage: $(basename "$0") [options] CLIP NAME
  -u URL     voice-clone service (default: \$QWEN_CLONE_URL or http://127.0.0.1:8011)
  -l LANG    language of the clip: de, en, fr, es, it, pt, ru, ja, ko, zh (required)
  -t TEXT    exact transcript of CLIP (default: auto-transcribe and trim)
  -s SECS    max reference length when auto-trimming, 3-30 (default: 12)
  -p TEXT    preview sentence (default: a stock sentence for de/en/fr)
  -n         no preview (skips the slowest step)
  -f         overwrite an existing voice of that name
  -o DIR     where to download the results (default: ./voices; user@host:path is scp'd)
NAME: lowercase [a-z0-9_-], becomes the TTS 'voice' value.
EOF
  exit 2
}

url="${QWEN_CLONE_URL:-http://127.0.0.1:8011}"
lang="" text="" max_ref="" preview_text="" preview="true" overwrite="false" out_dir="./voices"
while getopts "u:l:t:s:p:nfo:h" opt; do
  case "$opt" in
    u) url="$OPTARG" ;;
    l) lang="$OPTARG" ;;
    t) text="$OPTARG" ;;
    s) max_ref="$OPTARG" ;;
    p) preview_text="$OPTARG" ;;
    n) preview="false" ;;
    f) overwrite="true" ;;
    o) out_dir="$OPTARG" ;;
    *) usage ;;
  esac
done
shift $((OPTIND - 1))
[[ $# -eq 2 && -n "$lang" ]] || usage
clip="$1" name="$2"
[[ -f "$clip" ]] || { echo "no such file: $clip" >&2; exit 1; }

form=(-F "file=@${clip}" -F "name=${name}" -F "language=${lang}" -F "preview=${preview}" -F "overwrite=${overwrite}")
[[ -n "$text" ]] && form+=(-F "ref_text=${text}")
[[ -n "$max_ref" ]] && form+=(-F "max_ref_seconds=${max_ref}")
[[ -n "$preview_text" ]] && form+=(-F "preview_text=${preview_text}")

echo "Uploading ${clip} to ${url} as voice '${name}' (${lang})..."
echo "(with a preview this takes a few seconds on GPU, a minute or more on CPU)"
response="$(mktemp)"
trap 'rm -f "$response"' EXIT
status="$(curl -sS -o "$response" -w '%{http_code}' --max-time 1800 "${form[@]}" "${url%/}/v1/voices")"
python3 -m json.tool "$response" 2>/dev/null || cat "$response"
echo
if [[ "$status" != "200" ]]; then
  echo "clone failed (HTTP ${status})" >&2
  exit 1
fi

exec "$(dirname "${BASH_SOURCE[0]}")/fetch_voice.sh" -u "$url" -o "$out_dir" "$name"
