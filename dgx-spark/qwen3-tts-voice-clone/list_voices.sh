#!/usr/bin/env bash
set -euo pipefail

# List the voices the voice-clone service has made (name, language, model,
# reference length, transcript).
#   ./list_voices.sh [-u URL]

url="${QWEN_CLONE_URL:-http://127.0.0.1:8011}"
while getopts "u:h" opt; do
  case "$opt" in
    u) url="$OPTARG" ;;
    *) echo "usage: $(basename "$0") [-u URL]" >&2; exit 2 ;;
  esac
done

json="$(curl -fsS "${url%/}/v1/voices")"
python3 - "$json" <<'EOF'
import json, sys
voices = json.loads(sys.argv[1])["voices"]
if not voices:
    print("no voices yet")
for name, meta in voices.items():
    if "error" in meta:
        print(f"{name}: unreadable ({meta['error']})")
        continue
    print(f"{name}  [{meta.get('language')}]  {meta.get('ref_seconds')}s ref  "
          f"{meta.get('base_model')}  {meta.get('created')}")
    print(f"    \"{meta.get('ref_text', '')}\"")
EOF
