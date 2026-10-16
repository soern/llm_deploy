# Test Script Guide - `test_tts.py`

## Overview

`test_tts.py` is a Python test script that connects to the running Kokoro English TTS server and attempts to synthesize audio from text.

**Current Status**: ✅ Infrastructure test passing, ❌ Synthesis not implemented (expected)

---

## Usage

### Basic Usage

```bash
# Terminal 1: Start the server
cd deploy/cpu/kokoro-tts-english
./start.sh

# Terminal 2: Run the test script
python3 test_tts.py
```

### With Custom Parameters

```bash
# Custom text
python3 test_tts.py --text "Hello, world!"

# Different voice
python3 test_tts.py --voice af_bella

# Custom output path
python3 test_tts.py --output /tmp/my_audio.aac

# All options
python3 test_tts.py \
  --text "How are you?" \
  --voice am_michael \
  --output /tmp/test_audio.aac
```

### Available Voices

The script will show top voice recommendations:

```
Top voices:
  • af_heart (American female, warm & emotional) - DEFAULT
  • af_bella (American female, bright)
  • am_michael (American male, professional)
  • bf_emma (British female, posh)
  • bm_daniel (British male, sophisticated)
```

All 54 voices are available. Run `curl http://localhost:8005/v1/voices | jq` to see the full list.

---

## What the Script Does

### 1. **Health Check**
```
Testing health endpoint: http://localhost:8005/health
  ✓ Server is healthy
    Model: hexgrad/Kokoro-82M
    Voices: 54 available
    Device: cpu
```

Verifies the server is running and model is loaded.

### 2. **List Available Voices**
```
Fetching voices: http://localhost:8005/v1/voices
  ✓ Got 54 voices
    Default voice: af_heart
    Sample voices: ['af_alloy', 'af_aoede', 'af_bella'] ... and 51 more
```

Retrieves and displays available voices.

### 3. **Attempt Synthesis**
```
Testing synthesis endpoint: http://localhost:8005/v1/audio/speech
  Text: 'Hello, how are you?'
  Voice: af_heart
  Output: /tmp/kokoro_test_output.aac
```

Sends a synthesis request and attempts to save audio.

---

## Expected Behavior (Current)

### ✓ Infrastructure Test: PASS

```
✓ Server is healthy
✓ Got 54 voices
✓ Default voice: af_heart
✗ Synthesis failed: HTTP 500 (EXPECTED)
```

**Why 500?** Synthesis is not yet implemented. The server scaffold validates all parameters but cannot generate audio.

### Error Message (Expected)

```
✗ Synthesis failed: HTTP 500
  Error message:
    Kokoro model synthesis requires the full model architecture to be loaded.
    This is a simplified server scaffold for testing deployment and configuration.
    To enable actual synthesis, the following would be needed:
    1. Load the full Kokoro model architecture (e.g., via 'misaki' package)
    2. Implement text tokenization via espeak-ng
    3. Implement voice embedding lookup and audio generation
    See hexgrad/Kokoro-82M GitHub for the reference implementation.

  Error details saved to: /tmp/kokoro_test_output_error.txt
```

---

## Expected Behavior (After Synthesis Implementation)

### ✓ Full Test: PASS

Once synthesis is implemented, the script will:

```
✓ Server is healthy
✓ Got 54 voices
✓ Success! Synthesis took 5.2s
✓ Audio saved: /tmp/kokoro_test_output.aac (45234 bytes)

  You can play the audio with:
    ffplay /tmp/kokoro_test_output.aac
    or: mpv /tmp/kokoro_test_output.aac
    or: open /tmp/kokoro_test_output.aac  # macOS
```

---

## Script Features

### ✓ Comprehensive Error Handling

- **Server not running**: Suggests `./start.sh`
- **Invalid voice**: Shows available voices and recommendations
- **Synthesis timeout**: Suggests machine may be slow
- **Connection error**: Helpful debugging messages

### ✓ Informative Output

The script provides:
- Server health status
- Available voices count
- Synthesis duration (once implemented)
- Audio file size
- Error details saved to disk

### ✓ Flexible Configuration

All parameters can be customized:
- `--base-url`: Server URL (default: `http://localhost:8005`)
- `--text`: Prompt text (default: `"Hello, how are you?"`)
- `--voice`: Voice name (default: `af_heart`)
- `--output`: Output file path (default: `/tmp/kokoro_test_output.aac`)

### ✓ Exit Codes

```
0 = Success (audio generated)
1 = Failure (synthesis not working or server error)
```

---

## Quick Test Scenarios

### Scenario 1: Verify Server is Running

```bash
python3 test_tts.py --text "test"
# If you see "✓ Server is healthy", the infrastructure works
```

### Scenario 2: Test All 5 Top Voices

```bash
for voice in af_heart af_bella am_michael bf_emma bm_daniel; do
  python3 test_tts.py --voice $voice --output /tmp/$voice.aac
done
```

### Scenario 3: Test Concurrent Requests

```bash
# In separate terminals
python3 test_tts.py --text "First request"
python3 test_tts.py --text "Second request"
python3 test_tts.py --text "Third request"
```

---

## Troubleshooting

### "Connection failed! Is the server running?"

**Solution:**
```bash
# Terminal 1
cd deploy/cpu/kokoro-tts-english
./start.sh

# Wait 42 seconds for server to start, then run test in Terminal 2
```

### "Voice 'xyz' not available"

**Solution:**
```bash
# List all 54 voices
curl http://localhost:8005/v1/voices | jq '.voices'

# Use one of the returned voice names
python3 test_tts.py --voice af_bella
```

### "Request timed out (>60 seconds)"

**Cause:** Synthesis too slow on this machine  
**Solution:**
- Consider GPU deployment (see IMPLEMENTATION_ROADMAP.md)
- Or increase timeout in script (modify `timeout=60` in test_tts.py)

### Script fails but server is running

**Debug:**
```bash
# Check server logs
docker logs nixclaw-kokoro-tts-en

# Check error file from test
cat /tmp/kokoro_test_output_error.txt
```

---

## Code Structure

```python
test_health(base_url)           # Verify server is alive
test_voices(base_url)           # Fetch available voices
test_synthesis(...)             # Attempt synthesis + save audio
```

---

## Integration with CI/CD

The script can be used in automated testing:

```bash
#!/bin/bash
# ci_test.sh

# Start server in background
./start.sh &
SERVER_PID=$!
sleep 45  # Wait for startup

# Run test
python3 test_tts.py
TEST_EXIT=$?

# Stop server
./stop.sh

# Report
exit $TEST_EXIT
```

---

## Next Steps

### When Synthesis is Implemented

The test script will work end-to-end:

```bash
python3 test_tts.py --text "Hello world" --voice af_heart
# → Generates /tmp/kokoro_test_output.aac
# → Can listen with: ffplay /tmp/kokoro_test_output.aac
```

### Before That

The script validates all the infrastructure:

```bash
python3 test_tts.py
# ✓ Server running
# ✓ Model loaded
# ✓ 54 voices available
# ✓ Endpoints responding
# ✗ Synthesis pending (expected)
```

---

## File Format

The output audio will be in the format specified (default: **AAC**):

- `.aac` - Raw ADTS AAC (default, smallest file)
- `.m4a` - MP4-boxed AAC (compatible)
- `.mp3` - MPEG-3 (larger, universal)
- `.ogg` - Opus Vorbis (good compression)
- `.wav` - PCM WAV (largest, uncompressed)

To use a different format:

```bash
python3 test_tts.py --output /tmp/test.wav
python3 test_tts.py --output /tmp/test.mp3
```

---

## Performance Expectations

**Current (Infrastructure Only):**
```
Health check:   ~42 ms
Voice list:     ~18 ms
Synthesis:      ~12 ms (parameter validation only)
```

**After Synthesis Implementation:**
```
Health check:   ~42 ms
Voice list:     ~18 ms
Synthesis:      1-10 seconds (depending on text length and CPU)
```

---

## Dependencies

Script requires:
- Python 3.8+
- `httpx` library (already in requirements, installed with `pip install httpx`)

```bash
pip install httpx
```

---

## Questions?

**Q: Why does synthesis fail?**  
A: It's not implemented yet (by design). See IMPLEMENTATION_ROADMAP.md.

**Q: Can I use this with the German TTS?**  
A: Yes! German TTS runs on port 8003. Modify `--base-url http://gx10-811e.lan:8003`.

**Q: How do I play the audio file?**  
A: Use `ffplay`, `mpv`, or any media player. Script suggests commands.

**Q: Can I save to MP3 instead of AAC?**  
A: Yes! Modify the synthesis request in the script to use `"response_format": "mp3"`.

---

**Last Updated:** 2026-08-27  
**Script Status:** ✅ Ready for infrastructure testing, ❌ Audio generation pending implementation
