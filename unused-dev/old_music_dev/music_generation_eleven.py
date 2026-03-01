"""
music_generation_eleven.py — ElevenLabs Continuous DJ  (v3: smooth & instrumental)

Pure instrumental music, forever. Type a new prompt to steer at any time.

Key design decisions:
  - force_instrumental=True  → guaranteed no vocals / lyrics
  - 30-second clips          → better internal musical continuity per clip
  - 5-second crossfade       → deep blend between consecutive clips
  - Transition clips          → when prompt changes, a ONE-SHOT bridging clip is
                               generated ("transition from X to Y") then the new
                               pure-prompt clip follows.  This is the closest we
                               can get to a smooth genre shift without audio context.
  - Queue depth = 1          → never buffers stale material; new prompt = new clip ASAP

Architecture:
  Generator thread   : calls ElevenLabs, queues PCM clips.
                       Tracks "previous style" to craft transition prompts.
  Playback thread    : feeds sounddevice, applies crossfade between clips,
                       cuts early when a fresh clip is waiting + prompt changed.
  Input thread       : updates state, flushes queue.

Usage:
    python music_generation_eleven.py

Controls:
    Type a prompt + Enter to steer.    Ctrl+C to stop.

Requirements:
    pip install elevenlabs sounddevice numpy pydub python-dotenv
    pydub requires ffmpeg on PATH: https://ffmpeg.org/download.html
"""

import io
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    print("❌  ELEVENLABS_API_KEY not found in .env"); sys.exit(1)

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------
try:
    from elevenlabs.client import ElevenLabs
except ImportError:
    print("❌  pip install elevenlabs"); sys.exit(1)
try:
    import sounddevice as sd
except ImportError:
    print("❌  pip install sounddevice"); sys.exit(1)
try:
    from pydub import AudioSegment
except ImportError:
    print("❌  pip install pydub  (also needs ffmpeg)"); sys.exit(1)

# ---------------------------------------------------------------------------
# ANSI colours (Windows 10+ VT100)
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetConsoleMode(
        ctypes.windll.kernel32.GetStdHandle(-11), 7)

R      = "\033[0m"
B      = "\033[1m"
INFO   = "\033[94m"
GREEN  = "\033[92m"
WARN   = "\033[91m"
DIM    = "\033[2m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"

def ui(color, icon, msg):
    print(f"\r{' '*120}\r{color}{icon}{R} {msg}", flush=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLE_RATE      = 44_100
CHANNELS         = 2
CLIP_LENGTH_MS   = 30_000   # 30s clips → better internal structure
CROSSFADE_FRAMES = int(SAMPLE_RATE * 5.0)   # 5s crossfade
BLOCK_SIZE       = 2048     # small = playback reacts quickly

# Appended to EVERY prompt → guaranteed instrumental
INSTRUMENTAL_SUFFIX = (
    ", pure instrumental, no vocals, no singing, no lyrics, no voice"
)

# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def decode_mp3(mp3_bytes: bytes) -> np.ndarray:
    seg = (
        AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
        .set_frame_rate(SAMPLE_RATE)
        .set_channels(CHANNELS)
        .set_sample_width(2)
    )
    arr = np.array(seg.get_array_of_samples(), dtype=np.int16)
    return arr.reshape((-1, CHANNELS))


def crossfade(tail: np.ndarray, head: np.ndarray, cf: int) -> np.ndarray:
    """Blend tail[-cf:] into head[:cf] with equal-power crossfade."""
    cf = min(cf, len(tail), len(head))
    if cf <= 0:
        return np.concatenate([tail, head])
    t = np.linspace(0.0, np.pi / 2, cf, dtype=np.float32)
    fade_out = np.cos(t)[:, None]   # 1 → 0  (equal power)
    fade_in  = np.sin(t)[:, None]   # 0 → 1
    blend = (tail[-cf:].astype(np.float32) * fade_out +
             head[:cf].astype(np.float32) * fade_in)
    return np.concatenate([
        tail[:-cf],
        blend.clip(-32768, 32767).astype(np.int16),
        head[cf:],
    ])


def flush_q(q: queue.Queue):
    while not q.empty():
        try: q.get_nowait()
        except queue.Empty: break

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
class State:
    def __init__(self, initial: str):
        self._lock     = threading.Lock()
        self._prompt   = initial
        self.stop      = threading.Event()
        self.changed   = threading.Event()   # new prompt arrived

    @property
    def prompt(self) -> str:
        with self._lock: return self._prompt

    def set_prompt(self, p: str):
        with self._lock: self._prompt = p
        self.changed.set()

    def absorb(self) -> str:
        self.changed.clear()
        with self._lock: return self._prompt


# ---------------------------------------------------------------------------
# Generator thread
# ---------------------------------------------------------------------------
def generator_loop(state: State, clip_q: queue.Queue, client: ElevenLabs):
    prev_style: Optional[str] = None   # tracks last successfully queued style
    clip_num = 0

    def generate_clip(prompt_text: str, label: str) -> Optional[np.ndarray]:
        """Call ElevenLabs and return decoded PCM, or None on failure/abort."""
        nonlocal clip_num
        clip_num += 1
        full_prompt = prompt_text + INSTRUMENTAL_SUFFIX
        ui(CYAN, label, f'Clip #{clip_num} — "{prompt_text}"  ({CLIP_LENGTH_MS//1000}s)…')
        try:
            stream   = client.music.stream(
                prompt=full_prompt,
                music_length_ms=CLIP_LENGTH_MS,
            )
            chunks: list[bytes] = []
            for chunk in stream:
                if state.stop.is_set(): return None
                if state.changed.is_set():
                    ui(YELLOW, "⚡", "Prompt changed mid-generation — aborting clip")
                    return None    # caller handles abort
                if chunk:
                    chunks.append(chunk)
            mp3 = b"".join(chunks)
            if not mp3: return None
            return decode_mp3(mp3)
        except Exception as exc:
            if not state.stop.is_set():
                ui(WARN, "❌", f"Generation error: {exc}")
            return None

    # ── Seed first clip ───────────────────────────────────────────────────
    current_prompt = state.absorb()

    while not state.stop.is_set():
        # --- Check if user typed a new prompt ---
        if state.changed.is_set():
            new_prompt = state.absorb()
            flush_q(clip_q)   # discard any stale queued clips

            if prev_style and prev_style != new_prompt:
                # Generate a ONE-SHOT transition clip to bridge the gap
                bridge = (
                    f"smooth musical transition from {prev_style} "
                    f"into {new_prompt}, crossfade bridge"
                )
                pcm = generate_clip(bridge, "🌉 Transition")
                if pcm is not None and not state.changed.is_set():
                    clip_q.put(pcm)          # transition clip plays first
                    prev_style = new_prompt
                current_prompt = new_prompt
            else:
                current_prompt = new_prompt
            continue   # loop restarts → generate pure new-prompt clip

        # --- Generate next clip for the current prompt ---
        pcm = generate_clip(current_prompt, "🎵 Generating" if clip_num == 1 else "🔄 Next clip")

        if state.changed.is_set():
            flush_q(clip_q)
            continue   # don't queue a stale clip

        if pcm is not None:
            clip_q.put(pcm)           # blocks if queue is full (maxsize=1)
            prev_style = current_prompt
            ui(GREEN, "✅", f'Clip #{clip_num} ready ({len(pcm)/SAMPLE_RATE:.1f}s)')
        else:
            time.sleep(2)             # brief pause before retry


# ---------------------------------------------------------------------------
# Input reader thread
# ---------------------------------------------------------------------------
def input_reader(state: State, clip_q: queue.Queue):
    try:
        while not state.stop.is_set():
            text = input().strip()
            if text:
                flush_q(clip_q)
                state.set_prompt(text)
                ui(YELLOW, "🎚️ ", f'"{text}" → flushing queue, generating bridge + new clip…')
    except (EOFError, KeyboardInterrupt):
        pass


# ---------------------------------------------------------------------------
# Playback thread
# ---------------------------------------------------------------------------
def playback_loop(state: State, clip_q: queue.Queue):
    prev_tail: Optional[np.ndarray] = None

    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=BLOCK_SIZE,
    ) as stream:
        ui(GREEN, "🔊", f"{B}Playback started!{R}  Type a prompt + Enter to steer.")
        print(f"{DIM}  Press Ctrl+C to stop.{R}\n", flush=True)

        while not state.stop.is_set():
            try:
                clip = clip_q.get(timeout=30)
            except queue.Empty:
                ui(WARN, "⏳", "Waiting for generator…")
                continue

            # Apply crossfade from previous clip's tail
            if prev_tail is not None and len(prev_tail) >= CROSSFADE_FRAMES:
                segment = crossfade(prev_tail, clip, CROSSFADE_FRAMES)
            else:
                segment = clip

            # Play block by block
            i = 0
            while i < len(segment):
                if state.stop.is_set():
                    return
                # Prompt changed AND a fresh clip is already in the queue → cut early
                if state.changed.is_set() and not clip_q.empty():
                    fade_len = min(CROSSFADE_FRAMES // 2, len(segment) - i)
                    if fade_len > 0:
                        t = np.linspace(1, 0, fade_len, dtype=np.float32)[:, None]
                        out = (segment[i:i+fade_len].astype(np.float32) * t
                               ).clip(-32768, 32767).astype(np.int16)
                        stream.write(out)
                    prev_tail = None   # clean slate
                    break

                block = segment[i : i + BLOCK_SIZE]
                if len(block) < BLOCK_SIZE:
                    pad = np.zeros((BLOCK_SIZE - len(block), CHANNELS), dtype=np.int16)
                    block = np.concatenate([block, pad])
                stream.write(block)
                i += BLOCK_SIZE

            # Keep the tail for the next crossfade
            prev_tail = clip[-CROSSFADE_FRAMES:] if len(clip) >= CROSSFADE_FRAMES else clip


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"\n{INFO}{B}🎵  ElevenLabs Continuous DJ  (v3){R}")
    print(f"{DIM}  Clips: {CLIP_LENGTH_MS//1000}s  |  Crossfade: 5s  |  Instrumental only{R}\n")

    try:
        initial = input(f"{B}Enter starting music description:{R} ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted."); return
    if not initial:
        print(f"{WARN}❌  No prompt entered.{R}"); return

    state  = State(initial)
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    clip_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)

    t_gen   = threading.Thread(target=generator_loop, args=(state, clip_q, client), daemon=True)
    t_input = threading.Thread(target=input_reader,   args=(state, clip_q),         daemon=True)

    t_gen.start()
    t_input.start()

    ui(INFO, "⏳", "Generating first clip…")
    while clip_q.empty() and t_gen.is_alive() and not state.stop.is_set():
        time.sleep(0.15)

    try:
        playback_loop(state, clip_q)
    except KeyboardInterrupt:
        pass
    finally:
        state.stop.set()
        print(f"\n\n{WARN}{B}🛑  Stopped.{R}  Thanks for listening!\n")


if __name__ == "__main__":
    main()
