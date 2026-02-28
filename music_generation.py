"""
music_generation.py

Generates ~6 seconds of instrumental music from a text prompt
using the Google Lyria RealTime API (lyria-realtime-exp model).

Usage:
    python music_generation.py "epic orchestral battle music"

Output:
    output_music.wav  (stereo, 48 kHz, 16-bit PCM)
"""

import asyncio
import os
import sys
import struct
import wave
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not found in .env")

MODEL = "models/lyria-realtime-exp"
SAMPLE_RATE = 48_000          # Hz  (spec: 48 kHz)
CHANNELS = 2                  # stereo
SAMPLE_WIDTH = 2              # bytes — 16-bit PCM
DURATION_SECONDS = 6
MAX_BYTES = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH * DURATION_SECONDS

OUTPUT_FILE = "output_music.wav"

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

client = genai.Client(
    api_key=GOOGLE_API_KEY,
    http_options={"api_version": "v1alpha"},
)


def save_wav(pcm_bytes: bytes, path: str) -> None:
    """Write raw 16-bit stereo PCM bytes to a .wav file."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)
    print(f"\n✅  Saved {len(pcm_bytes) / 1024:.1f} KB of audio → {path}")


async def generate_music(prompt: str) -> None:
    print(f"🎵  Generating {DURATION_SECONDS}s of music for prompt: '{prompt}'")
    print("    Connecting to Lyria RealTime …")

    audio_buffer = bytearray()

    async def receive_audio(session) -> None:
        """Collect audio chunks until we have enough data."""
        async for message in session.receive():
            if not message.server_content:
                continue
            chunks = message.server_content.audio_chunks
            if not chunks:
                continue
            audio_buffer.extend(chunks[0].data)
            received_secs = len(audio_buffer) / (SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH)
            print(
                f"    Received {received_secs:.2f}s / {DURATION_SECONDS}s …",
                end="\r",
                flush=True,
            )
            if len(audio_buffer) >= MAX_BYTES:
                return  # signal to stop

    async with (
        client.aio.live.music.connect(model=MODEL) as session,
        asyncio.TaskGroup() as tg,
    ):
        # Launch receiver task
        recv_task = tg.create_task(receive_audio(session))

        # Set prompt
        await session.set_weighted_prompts(
            prompts=[
                types.WeightedPrompt(text=prompt, weight=1.0),
            ]
        )

        # Set generation config
        await session.set_music_generation_config(
            config=types.LiveMusicGenerationConfig(
                bpm=120,
                temperature=1.0,
            )
        )

        # Start streaming
        await session.play()

        # Wait until we have 6 seconds worth of audio, then cancel
        while len(audio_buffer) < MAX_BYTES:
            await asyncio.sleep(0.05)

        recv_task.cancel()

    # Trim to exactly 6 seconds
    pcm = bytes(audio_buffer[:MAX_BYTES])
    save_wav(pcm, OUTPUT_FILE)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python music_generation.py \"<your music prompt>\"")
        sys.exit(1)

    prompt = " ".join(sys.argv[1:])
    asyncio.run(generate_music(prompt))


if __name__ == "__main__":
    main()
