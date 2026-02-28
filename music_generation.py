"""
music_generation.py  —  Live AI DJ powered by Google Lyria RealTime

Continuously plays AI-generated music through your speakers and lets you
steer the style in real-time by typing new prompts into the terminal.

Usage:
    python music_generation.py "chill lo-fi beats"

While running:
    • Type a new prompt and press Enter to smoothly transition the music
    • Press Ctrl+C to stop

Architecture:
    One persistent WebSocket connection to Lyria RealTime.
    Only sends API messages when the prompt actually changes.
    Audio is streamed in real-time through sounddevice (PortAudio).
"""

import asyncio
import collections
import os
import sys
import threading
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

try:
    import sounddevice as sd
except ImportError:
    print("❌  sounddevice is required:  pip install sounddevice")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not found in .env")

MODEL = "models/lyria-realtime-exp"
SAMPLE_RATE = 48_000       # Hz  (Lyria spec)
CHANNELS = 2               # stereo
SAMPLE_WIDTH = 2            # 16-bit PCM → 2 bytes per sample
BYTES_PER_FRAME = CHANNELS * SAMPLE_WIDTH   # 4 bytes per frame
BLOCK_SIZE = 2400           # frames per sounddevice callback (50 ms @ 48 kHz)

# Pre-buffer: accumulate this many seconds of audio before starting playback
# to avoid underruns while the WebSocket fills the buffer.
PRE_BUFFER_SECONDS = 0.5
PRE_BUFFER_BYTES = int(PRE_BUFFER_SECONDS * SAMPLE_RATE * BYTES_PER_FRAME)

# Ring-buffer capacity in bytes (~10 seconds of audio)
BUFFER_CAPACITY = SAMPLE_RATE * BYTES_PER_FRAME * 10

client = genai.Client(
    api_key=GOOGLE_API_KEY,
    http_options={"api_version": "v1alpha"},
)


# ---------------------------------------------------------------------------
# Thread-safe audio ring buffer
# ---------------------------------------------------------------------------
class AudioBuffer:
    """A simple thread-safe byte buffer backed by a deque of chunks."""

    def __init__(self, capacity: int = BUFFER_CAPACITY):
        self._chunks: collections.deque[bytes] = collections.deque()
        self._total_bytes = 0
        self._capacity = capacity
        self._lock = threading.Lock()
        self._total_received = 0

    def push(self, data: bytes) -> None:
        with self._lock:
            self._chunks.append(data)
            self._total_bytes += len(data)
            self._total_received += len(data)
            # Evict oldest if over capacity
            while self._total_bytes > self._capacity and self._chunks:
                old = self._chunks.popleft()
                self._total_bytes -= len(old)

    def pull(self, n: int) -> bytes:
        """Pull up to n bytes from the buffer. Returns silence if empty."""
        with self._lock:
            if self._total_bytes == 0:
                return b"\x00" * n

            result = bytearray()
            while len(result) < n and self._chunks:
                chunk = self._chunks[0]
                needed = n - len(result)
                if len(chunk) <= needed:
                    result.extend(chunk)
                    self._total_bytes -= len(chunk)
                    self._chunks.popleft()
                else:
                    result.extend(chunk[:needed])
                    self._chunks[0] = chunk[needed:]
                    self._total_bytes -= needed

            # Pad with silence if not enough data
            if len(result) < n:
                result.extend(b"\x00" * (n - len(result)))
            return bytes(result)

    @property
    def available(self) -> int:
        with self._lock:
            return self._total_bytes

    @property
    def received_seconds(self) -> float:
        with self._lock:
            return self._total_received / (SAMPLE_RATE * BYTES_PER_FRAME)


# ---------------------------------------------------------------------------
# Terminal input reader (runs in a background thread)
# ---------------------------------------------------------------------------
def input_reader(prompt_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    """Blocking thread that reads lines from stdin and pushes them to the async queue."""
    try:
        while True:
            line = input()
            text = line.strip()
            if text:
                loop.call_soon_threadsafe(prompt_queue.put_nowait, text)
    except (EOFError, KeyboardInterrupt):
        pass


# ---------------------------------------------------------------------------
# Main async logic
# ---------------------------------------------------------------------------
async def live_dj(initial_prompt: str) -> None:
    audio_buf = AudioBuffer()
    prompt_queue: asyncio.Queue[str] = asyncio.Queue()
    current_prompt = initial_prompt
    playback_started = threading.Event()
    stop_event = asyncio.Event()

    # --- Sounddevice callback (runs in audio thread) ---
    def audio_callback(outdata, frames, time_info, status):
        n_bytes = frames * BYTES_PER_FRAME
        data = audio_buf.pull(n_bytes)
        outdata[:] = memoryview(data).cast("B").cast("h", shape=(frames, CHANNELS))  # noqa: E501

    # --- Task: receive audio chunks from Lyria ---
    async def receive_audio(session):
        buffering = True
        async for message in session.receive():
            if stop_event.is_set():
                return
            if not message.server_content or not message.server_content.audio_chunks:
                continue
            for chunk in message.server_content.audio_chunks:
                audio_buf.push(chunk.data)

            # Start playback once we have enough pre-buffer
            if buffering and audio_buf.available >= PRE_BUFFER_BYTES:
                buffering = False
                playback_started.set()

    # --- Task: watch for new prompts and send to Lyria ---
    async def prompt_handler(session):
        nonlocal current_prompt
        while not stop_event.is_set():
            try:
                new_prompt = await asyncio.wait_for(prompt_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            if new_prompt != current_prompt:
                current_prompt = new_prompt
                print(f"\n🎶  Transitioning to: \"{current_prompt}\"")
                await session.set_weighted_prompts(
                    prompts=[
                        types.WeightedPrompt(text=current_prompt, weight=1.0),
                    ]
                )

    # --- Task: status display ---
    async def status_display():
        start_time = time.time()
        while not stop_event.is_set():
            elapsed = time.time() - start_time
            buf_secs = audio_buf.available / (SAMPLE_RATE * BYTES_PER_FRAME)
            mins, secs = divmod(int(elapsed), 60)
            print(
                f"  ▶ Playing [{mins:02d}:{secs:02d}]  "
                f"Buffer: {buf_secs:.1f}s  "
                f"Prompt: \"{current_prompt}\"    ",
                end="\r",
                flush=True,
            )
            await asyncio.sleep(1.0)

    # --- Launch everything ---
    print(f"🎵  Live AI DJ — Starting with: \"{initial_prompt}\"")
    print("    Connecting to Lyria RealTime …\n")

    loop = asyncio.get_event_loop()

    # Start terminal input reader thread
    input_thread = threading.Thread(
        target=input_reader, args=(prompt_queue, loop), daemon=True
    )
    input_thread.start()

    try:
        async with (
            client.aio.live.music.connect(model=MODEL) as session,
            asyncio.TaskGroup() as tg,
        ):
            # Set initial prompt
            await session.set_weighted_prompts(
                prompts=[
                    types.WeightedPrompt(text=initial_prompt, weight=1.0),
                ]
            )

            # Set generation config — QUALITY mode for best results
            await session.set_music_generation_config(
                config=types.LiveMusicGenerationConfig(
                    temperature=1.0,
                )
            )

            # Start streaming from Lyria
            await session.play()

            # Launch async tasks
            tg.create_task(receive_audio(session))
            tg.create_task(prompt_handler(session))
            tg.create_task(status_display())

            # Wait for pre-buffer to fill before starting audio output
            print("    Buffering …", end="\r", flush=True)
            while not playback_started.is_set():
                await asyncio.sleep(0.05)

            print("    🔊 Playback started! Type a new prompt and press Enter to steer the music.")
            print("    Press Ctrl+C to stop.\n")

            # Open sounddevice stream and keep it running
            with sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=BLOCK_SIZE,
                callback=audio_callback,
            ):
                # Run forever until Ctrl+C
                while True:
                    await asyncio.sleep(0.5)

    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        stop_event.set()
        print("\n\n🛑  Stopped. Thanks for listening!")


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python music_generation.py "<initial music prompt>"')
        print('Example: python music_generation.py "chill lo-fi beats"')
        sys.exit(1)

    prompt = " ".join(sys.argv[1:])
    try:
        asyncio.run(live_dj(prompt))
    except KeyboardInterrupt:
        print("\n🛑  Stopped. Thanks for listening!")


if __name__ == "__main__":
    main()
