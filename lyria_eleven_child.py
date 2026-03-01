"""
lyria_eleven_child.py — ElevenLabs (fast metadata) → Lyria Live DJ

CONCEPT
-------
When the user types a prompt, we open an ElevenLabs compose_detailed()
HTTP stream and read chunks ONLY until the JSON metadata appears in the
multipart response. We then close the connection immediately, before
ElevenLabs has generated any audio. The rich compositionPlan is converted
into Lyria WeightedPrompts and pushed to a persistent Lyria WebSocket.

    User prompt
        → ElevenLabs multipart stream (JSON part only, connection closed early)
        → Lyria WeightedPrompts
        → live Lyria audio

Why this is faster than compose_detailed():
    The multipart response sends JSON FIRST, then audio.
    We close the stream as soon as we have the JSON,
    skipping audio generation + download entirely.

USAGE
-----
    python lyria_eleven_child.py "medieval tavern"
    python lyria_eleven_child.py "dark forest" --temperature 1.1 --guidance 3.5

WHILE RUNNING
-------------
    Type a new description + Enter → ElevenLabs metadata → Lyria steers
    Ctrl+C → stop

REQUIREMENTS
------------
    pip install google-genai elevenlabs python-dotenv sounddevice
"""

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from typing import Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config / env
# ---------------------------------------------------------------------------
load_dotenv()

if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetConsoleMode(ctypes.windll.kernel32.GetStdHandle(-11), 7)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not found in .env")

# ---------------------------------------------------------------------------
# Dependency imports
# ---------------------------------------------------------------------------
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("❌  pip install google-genai"); sys.exit(1)

# ElevenLabs is imported lazily inside eleven_to_palette() — only needed with --eleven

try:
    import sounddevice as sd
except ImportError:
    print("❌  pip install sounddevice"); sys.exit(1)

import collections

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    STATUS = "\033[90m"
    VOICE  = "\033[96m"
    THINK  = "\033[93m"
    APPLIED= "\033[92m"
    WARN   = "\033[91m"
    INFO   = "\033[94m"
    DIM    = "\033[2m"

def ui_print(color, prefix, msg, **kw):
    print(f"\r{' '*110}\r{color}{prefix}{C.RESET} {msg}", **kw)

# ---------------------------------------------------------------------------
# Lyria constants
# ---------------------------------------------------------------------------
LYRIA_MODEL  = "models/lyria-realtime-exp"
SAMPLE_RATE  = 48_000
CHANNELS     = 2
SAMPLE_WIDTH = 2
BYTES_PER_FRAME = CHANNELS * SAMPLE_WIDTH
BLOCK_SIZE   = 2400
PRE_BUFFER_BYTES = int(0.5 * SAMPLE_RATE * BYTES_PER_FRAME)
BUFFER_CAPACITY  = SAMPLE_RATE * BYTES_PER_FRAME * 10

lyria_client  = genai.Client(
    api_key=GOOGLE_API_KEY,
    http_options={"api_version": "v1alpha"},
)

# ---------------------------------------------------------------------------
# ElevenLabs → Lyria palette converter
# ---------------------------------------------------------------------------
def eleven_to_palette(prompt: str) -> dict[str, float]:
    """Call ElevenLabs compose_detailed() for rich musical DNA. Only used with --eleven."""
    try:
        from elevenlabs.client import ElevenLabs as _ElevenLabs
    except ImportError:
        ui_print(C.WARN, "⚠️ ", "elevenlabs not installed — pip install elevenlabs")
        return {prompt: 1.0}
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        ui_print(C.WARN, "⚠️ ", "ELEVENLABS_API_KEY not found in .env")
        return {prompt: 1.0}
    client = _ElevenLabs(api_key=api_key)
    ui_print(C.THINK, "🔬", f'ElevenLabs palette for "{prompt}"…', flush=True)
    try:
        response = client.music.compose_detailed(
            prompt=prompt,
            music_length_ms=3_000,
            force_instrumental=True,
            output_format="mp3_22050_32",
        )
    except Exception as exc:
        ui_print(C.WARN, "⚠️ ", f"ElevenLabs failed ({exc}) — fallback")
        return {prompt: 1.0}

    meta    = response.json
    palette: dict[str, float] = {}

    plan = meta.get("composition_plan", {})
    for style in plan.get("positive_global_styles", []):
        text = str(style).strip()
        if text:
            palette[text] = 1.2

    sections = plan.get("sections", [])
    if sections:
        for style in sections[0].get("positive_local_styles", []):
            text = str(style).strip()
            if text and text not in palette:
                palette[text] = 0.9

    song_meta = meta.get("song_metadata", {})
    for genre in song_meta.get("genres", []):
        text = str(genre).strip()
        if text and text not in palette:
            palette[text] = 1.0

    desc = song_meta.get("description", "")
    if desc:
        for part in str(desc).split(","):
            text = part.strip()
            if text and len(text) < 60 and text not in palette:
                palette[text] = 0.6

    if not palette:
        ui_print(C.WARN, "⚠️ ", "Empty palette — fallback")
        return {prompt: 1.0}

    return palette


def raw_to_palette(prompt: str) -> dict[str, float]:
    """Pass the prompt straight to Lyria as-is — no ElevenLabs call."""
    return {prompt: 1.0}


def palette_display(palette: dict[str, float]) -> str:
    parts = [f'"{t}":{w:.2f}' for t, w in sorted(palette.items(), key=lambda x: -x[1])]
    return "  ".join(parts)


def palette_to_weighted_prompts(palette: dict[str, float]) -> list[types.WeightedPrompt]:
    return [types.WeightedPrompt(text=t, weight=w) for t, w in palette.items()]


def interpolate_palettes(src: dict[str, float], dst: dict[str, float], t: float) -> dict[str, float]:
    """Linear crossfade between two palettes at position t (0→1)."""
    all_keys = set(src) | set(dst)
    result = {}
    for k in all_keys:
        w = src.get(k, 0.0) * (1 - t) + dst.get(k, 0.0) * t
        if w > 0.01:
            result[k] = round(w, 3)
    return result


# ---------------------------------------------------------------------------
# Ring buffer for Lyria audio
# ---------------------------------------------------------------------------
class AudioBuffer:
    def __init__(self, capacity=BUFFER_CAPACITY):
        self._chunks: collections.deque[bytes] = collections.deque()
        self._total = 0
        self._capacity = capacity
        self._lock = threading.Lock()

    def push(self, data: bytes):
        with self._lock:
            self._chunks.append(data)
            self._total += len(data)
            while self._total > self._capacity and self._chunks:
                old = self._chunks.popleft()
                self._total -= len(old)

    def pull(self, n: int) -> bytes:
        with self._lock:
            if self._total == 0:
                return b"\x00" * n
            result = bytearray()
            while len(result) < n and self._chunks:
                chunk = self._chunks[0]
                needed = n - len(result)
                if len(chunk) <= needed:
                    result.extend(chunk)
                    self._total -= len(chunk)
                    self._chunks.popleft()
                else:
                    result.extend(chunk[:needed])
                    self._chunks[0] = chunk[needed:]
                    self._total -= needed
            if len(result) < n:
                result.extend(b"\x00" * (n - len(result)))
            return bytes(result)

    @property
    def available(self) -> int:
        with self._lock:
            return self._total


# ---------------------------------------------------------------------------
# Terminal input reader
# ---------------------------------------------------------------------------
def input_reader(prompt_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
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
async def live_dj(initial_prompt: str, args: argparse.Namespace) -> None:
    audio_buf = AudioBuffer()
    prompt_queue: asyncio.Queue[str] = asyncio.Queue()
    playback_started = threading.Event()
    stop_event = asyncio.Event()

    # ── Palette translator (respects --eleven flag) ──────────────────────
    def to_palette(p: str) -> dict[str, float]:
        return eleven_to_palette(p) if args.eleven else raw_to_palette(p)

    # ── Step 1: translate initial prompt ─────────────────────────────────
    loop = asyncio.get_event_loop()
    initial_palette = await loop.run_in_executor(None, to_palette, initial_prompt)
    ui_print(C.APPLIED, "🧬", f"Musical DNA: {palette_display(initial_palette)}")

    current_palette: dict[str, float] = dict(initial_palette)
    palette_lock = threading.Lock()

    def get_palette():
        with palette_lock:
            return dict(current_palette)

    def set_palette(p):
        nonlocal current_palette
        with palette_lock:
            current_palette = dict(p)

    # ── Sounddevice callback ──────────────────────────────────────────────
    def audio_callback(outdata, frames, time_info, status):
        n_bytes = frames * BYTES_PER_FRAME
        data = audio_buf.pull(n_bytes)
        outdata[:] = memoryview(data).cast("B").cast("h", shape=(frames, CHANNELS))

    # ── Receive audio from Lyria ──────────────────────────────────────────
    async def receive_audio(session):
        buffering = True
        async for message in session.receive():
            if stop_event.is_set():
                return
            if not message.server_content or not message.server_content.audio_chunks:
                continue
            for chunk in message.server_content.audio_chunks:
                audio_buf.push(chunk.data)
            if buffering and audio_buf.available >= PRE_BUFFER_BYTES:
                buffering = False
                playback_started.set()

    # ── Steering handler ───────────────────────────────────────────────────
    async def steering_handler(session):
        while not stop_event.is_set():
            try:
                user_input = await asyncio.wait_for(prompt_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            ui_print(C.THINK, "🔬", f'ElevenLabs fast-metadata for "{user_input}"…', flush=True)
            t0 = time.perf_counter()

            src_palette = get_palette()

            # Run in thread pool — non-blocking for asyncio
            target = await loop.run_in_executor(None, to_palette, user_input)

            latency_ms = (time.perf_counter() - t0) * 1000
            ui_print(
                C.THINK, "🎚️ ",
                f"New palette → {palette_display(target)}  {C.DIM}[{latency_ms:.0f}ms]{C.RESET}",
            )

            # Crossfade from current → target over args.crossfade seconds
            if args.crossfade > 0:
                n_steps = max(2, args.steps)
                step_sleep = args.crossfade / n_steps
                for i in range(1, n_steps + 1):
                    if stop_event.is_set():
                        break
                    t = i / n_steps
                    mid = interpolate_palettes(src_palette, target, t)
                    set_palette(mid)
                    await session.set_weighted_prompts(prompts=palette_to_weighted_prompts(mid))
                    await asyncio.sleep(step_sleep)

            set_palette(target)
            await session.set_weighted_prompts(prompts=palette_to_weighted_prompts(target))
            ui_print(C.APPLIED, "✅", f"{C.BOLD}Applied!{C.RESET} Lyria steering: {palette_display(target)}")

    # ── Status display ─────────────────────────────────────────────────────
    async def status_display():
        start = time.time()
        while not stop_event.is_set():
            elapsed = time.time() - start
            buf_s = audio_buf.available / (SAMPLE_RATE * BYTES_PER_FRAME)
            mins, secs = divmod(int(elapsed), 60)
            pal = get_palette()
            line = (
                f"{C.STATUS}  ▶ [{mins:02d}:{secs:02d}]  Buf:{buf_s:.1f}s  "
                f"{C.DIM}Palette: {palette_display(pal)}{C.RESET}    "
            )
            print(line, end="\r", flush=True)
            await asyncio.sleep(1.0)

    # ── Launch ──────────────────────────────────────────────────────────────
    mode_label = "ElevenLabs DNA + Lyria" if args.eleven else "Lyria direct"
    print(f"\n{C.INFO}{C.BOLD}🎵  Lyria Live DJ  [{mode_label}]{C.RESET}")
    if args.eleven:
        print(f"{C.DIM}    ElevenLabs: compose_detailed() — rich musical DNA, audio discarded")
    else:
        print(f"{C.DIM}    Lyria direct: prompt used as-is (add --eleven for ElevenLabs DNA)")
    print(f"    Lyria     : live real-time audio generation")
    print(f"    Crossfade : {args.crossfade}s over {args.steps} steps")
    print(f"    Connecting to Lyria RealTime…{C.RESET}\n")

    _loop = asyncio.get_event_loop()
    t_input = threading.Thread(target=input_reader, args=(prompt_queue, _loop), daemon=True)
    t_input.start()

    try:
        async with (
            lyria_client.aio.live.music.connect(model=LYRIA_MODEL) as session,
            asyncio.TaskGroup() as tg,
        ):
            await session.set_weighted_prompts(
                prompts=palette_to_weighted_prompts(initial_palette)
            )
            await session.set_music_generation_config(
                config=types.LiveMusicGenerationConfig(
                    temperature=args.temperature,
                    guidance=args.guidance,
                    seed=args.seed,
                )
            )
            await session.play()

            tg.create_task(receive_audio(session))
            tg.create_task(steering_handler(session))
            tg.create_task(status_display())

            print(f"    {C.DIM}Buffering…{C.RESET}", end="\r", flush=True)
            while not playback_started.is_set():
                await asyncio.sleep(0.05)

            print(f"\n{C.APPLIED}{C.BOLD}    🔊 Playback started!{C.RESET}")
            print(f"{C.DIM}    Type a description and press Enter to steer the music.")
            print(f"    ElevenLabs translates prompt → Lyria steers the music.")
            print(f"    Press Ctrl+C to stop.{C.RESET}\n")

            with sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=BLOCK_SIZE,
                callback=audio_callback,
            ):
                while True:
                    await asyncio.sleep(0.5)

    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        stop_event.set()
        print(f"\n\n{C.WARN}{C.BOLD}🛑  Stopped.{C.RESET}  Thanks for listening!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Lyria × ElevenLabs Hybrid DJ — ElevenLabs extracts musical DNA, Lyria plays live"
    )
    parser.add_argument("prompt", nargs="+", help='Initial prompt e.g. "medieval tavern"')
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Lyria temperature 0.5–2.0 (default: 1.0)")
    parser.add_argument("--guidance", type=float, default=4,
                        help="Lyria guidance 1.0–6.0 (default: 3.0)")
    parser.add_argument("--crossfade", type=float, default=4.0,
                        help="Crossfade seconds between palette changes (default: 4.0)")
    parser.add_argument("--steps", type=int, default=8,
                        help="Crossfade interpolation steps (default: 8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Lyria generation seed for reproducibility (default: 42)")
    parser.add_argument("--eleven", action="store_true",
                        help="Use ElevenLabs to enrich prompts into musical DNA before sending to Lyria")
    args = parser.parse_args()

    initial_prompt = " ".join(args.prompt)
    asyncio.run(live_dj(initial_prompt, args))


if __name__ == "__main__":
    main()
