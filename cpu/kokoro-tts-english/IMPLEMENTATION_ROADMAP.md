# Kokoro English TTS - Implementation Roadmap

## Current Status: SCAFFOLD (Endpoints Ready, Synthesis Pending)

The server successfully:
- ✅ Downloads the model from HuggingFace
- ✅ Loads all 54 voice embeddings  
- ✅ Handles HTTP requests
- ✅ Validates parameters (text length, voice name, language, speed)
- ❌ Does NOT synthesize audio yet (this is the next step)

---

## What Needs to Happen

### Phase 1: Implement Synthesis (Estimated: 2-4 hours)

**Two options**, pick one:

#### Option A: Use `misaki` Package (RECOMMENDED)

```python
# In Dockerfile, add to pip install:
#   misaki

# In server.py, _load_model():
from misaki import Kokoro
_model = Kokoro.load_model("hexgrad/Kokoro-82M", device="cpu")

# In server.py, _synthesize_blocking():
try:
    # misaki API: model(text) -> np.ndarray
    samples = _model(text, voice=voice, speed=speed)
    if isinstance(samples, list):
        samples = np.array(samples)
    if samples.dtype != np.float32:
        samples = samples.astype(np.float32)
    return samples
except Exception as e:
    LOG.error(f"Synthesis failed: {e}")
    raise
```

**Pros:**
- Simplest (15-30 lines total)
- Already tested by hexgrad authors
- Most likely to work out-of-the-box

**Cons:**
- Dependency on `misaki` package maintenance
- Less control over phonemization

---

#### Option B: Raw PyTorch Forward Pass

```python
# In Dockerfile, keep existing dependencies (no misaki needed)

# In server.py, _load_model():
import torch
from phonemizer.backend import EspeakBackend

# Already loaded: _model_state (model weights), _voices (embeddings)
# Create tokenizer
_phonemizer = EspeakBackend(language='en')
_vocab = _build_vocab()  # Or load from config.json

# In server.py, _synthesize_blocking():
# 1. Tokenize text
tokens = _phonemizer.phonemize(text)
token_ids = _vocab.encode(tokens)
input_ids = torch.tensor([token_ids], device=_device)

# 2. Get voice embedding
voice_emb = _voices[voice]

# 3. Run model forward pass
with torch.no_grad():
    # Load model architecture from state_dict
    model = load_kokoro_architecture(_model_state, device=_device)
    # Forward pass (exact signature depends on Kokoro layer structure)
    output = model(input_ids, voice_emb, speed=speed)
    # Output is typically mel-spectrogram or raw samples
    samples = postprocess_mel_to_audio(output)

return samples.cpu().numpy()
```

**Pros:**
- Full control
- No external package dependencies
- Can optimize/quantize later

**Cons:**
- ~150-200 lines of code
- Requires understanding Kokoro model architecture
- More error-prone (phonemizer edge cases, tensor shapes)

---

### Testing Strategy

After implementing synthesis:

```bash
# 1. Basic test (shortest text)
curl -X POST http://localhost:8005/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hi.","voice":"af_heart"}' \
  -o /tmp/test.aac && file /tmp/test.aac

# 2. Measure latency
time curl -X POST http://localhost:8005/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello, how are you?","voice":"af_heart"}' \
  > /dev/null

# 3. Try different voices
for voice in af_heart af_bella am_michael bm_daniel bf_emma; do
  curl -X POST http://localhost:8005/v1/audio/speech \
    -d "{\"input\":\"Test voice $voice\",\"voice\":\"$voice\"}" \
    -o /tmp/$voice.aac
done

# 4. Test with German server running (parallel)
# Should show both languages working simultaneously
```

### Performance Expectations (Post-Implementation)

| Text | Length | Expected | Acceptable? |
|------|--------|----------|-------------|
| "Hi." | 3 tokens | 1-2s | ✅ Yes |
| "Hello world." | 5 tokens | 1-3s | ✅ Yes |
| "How are you?" | 5 tokens | 1-3s | ✅ Yes |
| 2-3 sentences | 50 tokens | 5-10s | ✅ Okay |
| Long paragraph | 200+ tokens | 20-30s | ⚠️ Slow |

---

### Phase 2: Integration with nixclaw Agent

Once synthesis works:

1. **Enable in config/agent.yaml**:
   ```yaml
   tts:
     enabled: true
     engines:
       en:
         base_url: "http://localhost:8005"
   ```

2. **Test text_to_speech tool** (already implemented):
   ```python
   # Tool is already in tools/__init__.py and tools/tts_ops.py
   # Just enable the engine and it should work
   from src.nixclaw_agent.tools.tts_ops import text_to_speech
   result = await text_to_speech("Hello!", language="en")
   # → returns filename like "speech_2026-08-27_12-00-00_a1b2.aac"
   ```

3. **Send via Signal** (agent already does this):
   ```python
   # Agent uses send_signal_attachment() to send the voice note
   # Automatically handles attachment path validation
   ```

4. **Test with mock Signal**:
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.test.yml up -d
   API_URL=http://localhost:8081 ./send_mock_message.sh +000000 "Say hello to me"
   docker logs nixclaw --tail 50 | grep -A 5 "text_to_speech"
   ```

---

### Phase 3: Optimization (Optional, If Needed)

Only do this if you encounter performance issues:

1. **Profile first**:
   ```bash
   python3 -c "
   import time
   start = time.time()
   # Call synthesize endpoint
   elapsed = time.time() - start
   print(f'Latency: {elapsed:.1f}s')
   "
   ```

2. **If CPU too slow** (>10s for short texts):
   - Option A: Deploy to DGX GPU (30 min rebuild)
   - Option B: Quantize model (int8 reduces latency ~30%, memory ~4x)
   - Option C: Cache synthesis results (same text → same audio)

3. **If memory tight** (>900 MB peak):
   - Use smaller model (if available)
   - Quantize model weights
   - Stream audio chunks instead of buffering full output

---

## Decision Tree

```
Start here: "I want to enable English TTS"
  │
  ├─→ "I want the simplest implementation"
  │     → Use Option A (misaki) + test locally
  │
  ├─→ "I want full control + minimal dependencies"
  │     → Use Option B (raw PyTorch) + understand Kokoro layers
  │
  └─→ "I'm not sure"
        → Start with Option A, switch to B if needed
        → (misaki is a good fallback strategy)
```

---

## File Locations

**Code to modify**:
- `deploy/cpu/kokoro-tts-english/server.py` - `_load_model()` and `_synthesize_blocking()`
- `Dockerfile` - Add `misaki` to pip install (if Option A)

**Tests to update**:
- Create `tests/test_kokoro_english_tts.py` (mock synthesis, measure latency)
- Update mock Signal E2E test to include English TTS

**Docs to update**:
- `README.md` - Remove "Synthesis not implemented" caveat
- `TEST_REPORT.md` - Add actual latency measurements

---

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|-----------|
| `misaki` package doesn't exist | Low | Use PyTorch forward pass instead |
| Phonemizer edge cases (abbreviations) | Medium | Test with German server logs for comparison |
| Model inference crashes | Medium | Add try/except, log full error, return 500 |
| Latency unacceptable on CPU | Medium | Profile early, plan GPU migration |
| Memory exhaustion | Low | Monitor peak memory, quantize if needed |
| Audio quality poor | Low | Try different voices, adjust speed |

---

## Success Criteria

After implementation, verify:

1. ✅ `curl http://localhost:8005/v1/audio/speech` returns binary audio
2. ✅ Audio file is valid AAC/MP4/WAV (use `file` command)
3. ✅ Audio plays in browser: `curl ... -o test.aac && mpv test.aac`
4. ✅ Latency acceptable (<10s for short texts)
5. ✅ Stress test: Multiple simultaneous requests handled
6. ✅ Integration test: Agent sends voice note to mock Signal
7. ✅ German TTS still works (both engines running)

---

## Estimated Timeline

| Task | Time | Notes |
|------|------|-------|
| Pick implementation option | 15 min | Read both + decide |
| Implement synthesis | 1-3 hours | Depends on option |
| Local testing | 30 min | Test all endpoints |
| Fix issues | 1-2 hours | Buffer for debugging |
| Integration test with agent | 1 hour | Mock Signal E2E |
| **Total** | **4-7 hours** | Realistic estimate |

---

## Questions?

- **"Will this work on my machine?"**: Almost certainly yes. If it fails, error messages are clear and Google-able.
- **"How do I know synthesis is working?"**: Listen to the output audio. Run `mpv /tmp/test.aac` or open in browser.
- **"What if it's too slow?"**: Profile first, then decide GPU vs quantization vs caching.
- **"Can I switch from Option A to B later?"**: Yes, completely independent implementations.

---

**Last Updated**: 2026-08-27  
**Next Milestone**: Complete synthesis implementation + mock Signal E2E test
