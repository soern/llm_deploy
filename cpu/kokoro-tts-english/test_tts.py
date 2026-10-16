#!/usr/bin/env python3
"""
Test script for Kokoro English TTS server.

Usage:
    # Start the server first:
    ./start.sh

    # In another terminal, run this script:
    python3 test_tts.py

    # Or with custom parameters:
    python3 test_tts.py --text "Hello, how are you?" --voice af_bella --output /tmp/test.aac
"""

import argparse
import sys
import time
from pathlib import Path

import httpx


def test_health(base_url: str) -> bool:
    """Test /health endpoint."""
    print(f"Testing health endpoint: {base_url}/health")
    try:
        response = httpx.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"  ✓ Server is healthy")
            print(f"    Model: {health.get('model')}")
            print(f"    Voices: {len(health.get('voices', []))} available")
            print(f"    Device: {health.get('device')}")
            return True
        else:
            print(f"  ✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_voices(base_url: str) -> list:
    """Get list of available voices."""
    print(f"\nFetching voices: {base_url}/v1/voices")
    try:
        response = httpx.get(f"{base_url}/v1/voices", timeout=5)
        if response.status_code == 200:
            data = response.json()
            voices = data.get("voices", [])
            print(f"  ✓ Got {len(voices)} voices")
            if len(voices) > 0:
                print(f"    Default voice: {data.get('default_voice')}")
                print(f"    Sample voices: {voices[:3]} ... and {len(voices)-3} more")
            return voices
        else:
            print(f"  ✗ Failed: {response.status_code}")
            return []
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return []


def test_synthesis(
    base_url: str,
    text: str,
    voice: str,
    output_path: str,
) -> bool:
    """Test /v1/audio/speech endpoint and save audio."""
    print(f"\nTesting synthesis endpoint: {base_url}/v1/audio/speech")
    print(f"  Text: {text!r}")
    print(f"  Voice: {voice}")
    print(f"  Output: {output_path}")

    try:
        start = time.time()
        response = httpx.post(
            f"{base_url}/v1/audio/speech",
            json={
                "input": text,
                "voice": voice,
                "language": "en",
                "response_format": "aac",
                "speed": 1.0,
            },
            timeout=60,  # Synthesis can be slow
        )
        elapsed = time.time() - start

        if response.status_code == 200:
            # Write audio file
            with open(output_path, "wb") as f:
                f.write(response.content)
            file_size = len(response.content)
            print(f"  ✓ Success! Synthesis took {elapsed:.1f}s")
            print(f"  ✓ Audio saved: {output_path} ({file_size} bytes)")
            print(f"\n  You can play the audio with:")
            print(f"    ffplay {output_path}")
            print(f"    or: mpv {output_path}")
            print(f"    or: open {output_path}  # macOS")
            return True
        else:
            # Likely NotImplementedError (synthesis not yet implemented)
            print(f"  ✗ Synthesis failed: HTTP {response.status_code}")
            error_msg = response.text[:500]  # First 500 chars of error
            print(f"  Error message:")
            for line in error_msg.split("\n"):
                if line.strip():
                    print(f"    {line}")
            
            # Save error response for debugging
            error_file = output_path.replace(".aac", "_error.txt")
            with open(error_file, "w") as f:
                f.write(f"HTTP {response.status_code}\n\n{error_msg}")
            print(f"\n  Error details saved to: {error_file}")
            return False

    except httpx.ConnectError:
        print(f"  ✗ Connection failed! Is the server running?")
        print(f"    Start with: ./start.sh")
        return False
    except httpx.TimeoutException:
        print(f"  ✗ Request timed out (>60 seconds)")
        print(f"    Synthesis may be too slow on this machine")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test Kokoro English TTS server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 test_tts.py
  python3 test_tts.py --text "Hello world" --voice af_heart
  python3 test_tts.py --text "How are you?" --voice am_michael --output /tmp/test.aac
        """,
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8005",
        help="TTS server base URL (default: http://localhost:8005)",
    )
    parser.add_argument(
        "--text",
        default="Hello, how are you?",
        help="Text to synthesize (default: 'Hello, how are you?')",
    )
    parser.add_argument(
        "--voice",
        default="af_heart",
        help="Voice to use (default: af_heart - warm, emotional female)",
    )
    parser.add_argument(
        "--output",
        default="/tmp/kokoro_test_output.aac",
        help="Output audio file path (default: /tmp/kokoro_test_output.aac)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("KOKORO ENGLISH TTS - TEST SCRIPT")
    print("=" * 80)

    # Test health
    if not test_health(args.base_url):
        print("\n✗ Server is not responding. Start it with: ./start.sh")
        return 1

    # Get available voices
    voices = test_voices(args.base_url)

    # Validate voice choice
    if args.voice not in voices:
        print(f"\n✗ Voice '{args.voice}' not available")
        if voices:
            print(f"  Available voices: {', '.join(voices[:5])} ... and {len(voices)-5} more")
            print(f"  Try one of these top voices:")
            print(f"    • af_heart (American female, warm & emotional) - DEFAULT")
            print(f"    • af_bella (American female, bright)")
            print(f"    • am_michael (American male, professional)")
            print(f"    • bf_emma (British female, posh)")
            print(f"    • bm_daniel (British male, sophisticated)")
        return 1

    # Test synthesis
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = test_synthesis(args.base_url, args.text, args.voice, str(output_path))

    print("\n" + "=" * 80)
    if success:
        print("✓ TEST PASSED")
        return 0
    else:
        print("✗ TEST FAILED")
        print("\nNote: If synthesis is not implemented yet, this is expected.")
        print("See IMPLEMENTATION_ROADMAP.md for next steps.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
