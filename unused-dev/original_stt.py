"""
stt.py — Speech-to-Text using Mistral's Voxtral API.

Usage:
  python stt.py            # Record 5 seconds, save recording.wav + transcription.txt
  python stt.py -live      # Stream mic in real-time, print transcription live (Ctrl+C to stop)

Requirements:
  pip install sounddevice scipy python-dotenv "mistralai[realtime]" pyaudio
"""

import os
import wave
import time
import asyncio
import argparse
import sounddevice as sd
from pathlib import Path
from dotenv import load_dotenv
from mistralai import Mistral

# ── Configuration ──────────────────────────────────────────────────────────────

RECORDING_DURATION  = 5
SAMPLE_RATE         = 16_000
CHANNELS            = 1
OUTPUT_AUDIO        = Path(__file__).parent / "recording.wav"
OUTPUT_TEXT         = Path(__file__).parent / "transcription.txt"
OFFLINE_MODEL       = "voxtral-mini-latest"
REALTIME_MODEL      = "voxtral-mini-transcribe-realtime-2602"

# ── Load API key ───────────────────────────────────────────────────────────────

load_dotenv(Path(__file__).parent / ".env")
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise EnvironmentError("MISTRAL_API_KEY not found. Check your .env file.")

# ══════════════════════════════════════════════════════════════════════════════
# DEFAULT MODE  — record 5 s then transcribe via offline API
# ══════════════════════════════════════════════════════════════════════════════

def record_audio(duration: int = RECORDING_DURATION, output_path: Path = OUTPUT_AUDIO) -> Path:
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

    with wave.open(str(output_path), "w") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())

    print(f"✅  Audio saved → {output_path}")
    return output_path


def transcribe(audio_path: Path = OUTPUT_AUDIO) -> str:
    client = Mistral(api_key=api_key)
    print(f"📡  Sending '{audio_path.name}' to Mistral API ({OFFLINE_MODEL})…")

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.complete(
            model=OFFLINE_MODEL,
            file={"content": f, "file_name": audio_path.name},
        )
    print()
    return response.text.strip()


def save_transcription(text: str, output_path: Path = OUTPUT_TEXT) -> None:
    output_path.write_text(text, encoding="utf-8")
    print(f"📄  Transcription saved → {output_path}")


def run_offline(
    *,
    duration: int = RECORDING_DURATION,
    output_audio: Path = OUTPUT_AUDIO,
    output_text: Path = OUTPUT_TEXT,
) -> None:
    audio_path = record_audio(duration=duration, output_path=output_audio)
    text = transcribe(audio_path)
    print("=" * 60)
    print("TRANSCRIPTION:")
    print(text)
    print("=" * 60)
    save_transcription(text, output_path=output_text)


# ══════════════════════════════════════════════════════════════════════════════
# LIVE MODE  — stream mic in real-time, print words as they come
# ══════════════════════════════════════════════════════════════════════════════

async def iter_microphone(*, sample_rate: int, chunk_duration_ms: int):
    """Yield raw PCM chunks from the microphone using PyAudio."""
    import pyaudio
    p = pyaudio.PyAudio()
    chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_samples,
    )
    loop = asyncio.get_running_loop()
    try:
        while True:
            data = await loop.run_in_executor(None, stream.read, chunk_samples, False)
            yield data
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


async def run_live():
    from mistralai.extra.realtime import UnknownRealtimeEvent
    from mistralai.models import (
        AudioFormat,
        RealtimeTranscriptionError,
        RealtimeTranscriptionSessionCreated,
        TranscriptionStreamDone,
        TranscriptionStreamTextDelta,
    )

    client = Mistral(api_key=api_key)
    audio_format = AudioFormat(encoding="pcm_s16le", sample_rate=SAMPLE_RATE)
    audio_stream = iter_microphone(sample_rate=SAMPLE_RATE, chunk_duration_ms=480)
    partial_text = ""

    print("\n🎙️  Live transcription started — speak freely. Press Ctrl+C to stop.\n")

    try:
        async for event in client.audio.realtime.transcribe_stream(
            audio_stream=audio_stream,
            model=REALTIME_MODEL,
            audio_format=audio_format,
        ):
            if isinstance(event, RealtimeTranscriptionSessionCreated):
                print("✅  Session connected. Listening…\n")
            elif isinstance(event, TranscriptionStreamTextDelta):
                partial_text += event.text
                print(event.text, end="", flush=True)
            elif isinstance(event, TranscriptionStreamDone):
                final_text = event.text.strip()
                if final_text:
                    if not partial_text.strip():
                        print(final_text, flush=True)
                    elif final_text != partial_text.strip():
                        print(f"\n{final_text}", flush=True)
                    else:
                        print(flush=True)
                else:
                    print(flush=True)
                partial_text = ""
            elif isinstance(event, RealtimeTranscriptionError):
                print(f"\n⚠️  Error: {event}")
            elif isinstance(event, UnknownRealtimeEvent):
                pass  # silently ignore unknown events
    except KeyboardInterrupt:
        print("\n\n⏹  Stopped.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Speech-to-text using Mistral's Voxtral API."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Stream microphone audio in real time instead of recording a fixed clip.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=RECORDING_DURATION,
        help="Recording duration in seconds for offline mode.",
    )
    parser.add_argument(
        "--output-audio",
        type=Path,
        default=OUTPUT_AUDIO,
        help="Path to the WAV file written in offline mode.",
    )
    parser.add_argument(
        "--output-text",
        type=Path,
        default=OUTPUT_TEXT,
        help="Path to the transcription text file written in offline mode.",
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()

    if args.live:
        asyncio.run(run_live())
    else:
        run_offline(
            duration=args.duration,
            output_audio=args.output_audio,
            output_text=args.output_text,
        )
