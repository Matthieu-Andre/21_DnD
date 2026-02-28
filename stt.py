"""
stt.py — Speech-to-Text using Mistral's Voxtral Mini (Realtime) via API.

Workflow:
  1. Record audio from the microphone (5 seconds by default)
  2. Save the recording as 'recording.wav'
  3. Send the audio to Mistral's transcription API (voxtral-mini-latest)
  4. Save the transcription to 'transcription.txt'

Requirements:
  pip install sounddevice scipy python-dotenv mistralai
"""

import os
import wave
import time
import tempfile
import numpy as np
import sounddevice as sd
from pathlib import Path
from dotenv import load_dotenv
from mistralai import Mistral

# ── Configuration ──────────────────────────────────────────────────────────────

RECORDING_DURATION = 5        # seconds to record
SAMPLE_RATE        = 16_000   # Hz  (16 kHz mono — optimal for speech models)
CHANNELS           = 1
OUTPUT_AUDIO       = Path(__file__).parent / "recording.wav"
OUTPUT_TEXT        = Path(__file__).parent / "transcription.txt"
MISTRAL_MODEL      = "voxtral-mini-latest"

# ── Load API key from .env ─────────────────────────────────────────────────────

load_dotenv(Path(__file__).parent / ".env")

api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise EnvironmentError(
        "MISTRAL_API_KEY not found. Make sure a .env file exists next to stt.py."
    )

# ── Step 1: Record from microphone ────────────────────────────────────────────

def record_audio(duration: int = RECORDING_DURATION, output_path: Path = OUTPUT_AUDIO) -> Path:
    """Record audio from the default microphone and save as a WAV file."""
    print(f"\n🎙️  Recording for {duration} seconds — speak now!")
    for i in range(3, 0, -1):
        print(f"   Starting in {i}...", flush=True)
        time.sleep(1)
    print("   ▶ Recording...", flush=True)

    audio_data = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
    )
    sd.wait()
    print("   ⏹  Done recording.\n")

    # Save as WAV
    with wave.open(str(output_path), "w") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)          # 16-bit = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())

    print(f"✅  Audio saved → {output_path}")
    return output_path


# ── Step 2: Transcribe via Mistral API ────────────────────────────────────────

def transcribe(audio_path: Path = OUTPUT_AUDIO) -> str:
    """Upload the audio file to Mistral's transcription endpoint and return the text."""
    client = Mistral(api_key=api_key)

    print(f"📡  Sending '{audio_path.name}' to Mistral API ({MISTRAL_MODEL})…")

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.complete(
            model=MISTRAL_MODEL,
            file={
                "content": f,
                "file_name": audio_path.name,
            },
        )

    text = response.text.strip()
    print(f"✅  Transcription received.\n")
    return text


# ── Step 3: Save transcription ────────────────────────────────────────────────

def save_transcription(text: str, output_path: Path = OUTPUT_TEXT) -> None:
    output_path.write_text(text, encoding="utf-8")
    print(f"📄  Transcription saved → {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    audio_path = record_audio()
    text       = transcribe(audio_path)

    print("=" * 60)
    print("TRANSCRIPTION:")
    print(text)
    print("=" * 60)

    save_transcription(text)


if __name__ == "__main__":
    main()
