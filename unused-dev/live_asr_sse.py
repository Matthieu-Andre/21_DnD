"""
Near-live microphone ASR using Mistral's streaming transcription endpoint.

This script uses the SSE endpoint documented at:
https://docs.mistral.ai/api/endpoint/audio/transcriptions#operation-audio_api_v1_transcriptions_post_stream

The endpoint accepts complete files, not an open-ended microphone stream.
To make it usable from the terminal in near-live mode, this script records
short microphone chunks, wraps each chunk as a WAV file, submits it with
diarization and timestamps enabled, and prints streamed updates as they arrive.

Example:
  python live_asr_sse.py
  python live_asr_sse.py --chunk-seconds 3.0 --language en
"""

from __future__ import annotations

import argparse
import io
import os
import queue
import shutil
import sys
import threading
import wave
from dataclasses import dataclass
from pathlib import Path

import httpx
import sounddevice as sd
from dotenv import load_dotenv
from mistralai.utils import eventstreaming, unmarshal_json
from mistralai.models import (
    TranscriptionStreamEvents,
    TranscriptionStreamDone,
    TranscriptionStreamLanguage,
    TranscriptionStreamSegmentDelta,
    TranscriptionStreamTextDelta,
)


DEFAULT_MODEL = "voxtral-mini-latest"
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH = 2
DEFAULT_CHUNK_SECONDS = 4.0
DEFAULT_READ_MS = 200
DEFAULT_QUEUE_SIZE = 8


@dataclass(frozen=True)
class ChunkJob:
    index: int
    start_sec: float
    pcm_bytes: bytes


class LivePrinter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._preview_len = 0

    def _truncate(self, text: str) -> str:
        width = shutil.get_terminal_size((100, 20)).columns
        if len(text) <= width - 1:
            return text
        return text[: max(0, width - 4)] + "..."

    def _clear_preview_locked(self) -> None:
        if self._preview_len:
            sys.stdout.write("\r" + (" " * self._preview_len) + "\r")
            sys.stdout.flush()
            self._preview_len = 0

    def preview(self, text: str) -> None:
        display = self._truncate(text)
        with self._lock:
            self._clear_preview_locked()
            sys.stdout.write(display)
            sys.stdout.flush()
            self._preview_len = len(display)

    def line(self, text: str = "") -> None:
        with self._lock:
            self._clear_preview_locked()
            print(text, flush=True)

    def clear_preview(self) -> None:
        with self._lock:
            self._clear_preview_locked()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Near-live diarized ASR using Mistral's SSE transcription endpoint."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Transcription model.")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Microphone sample rate in Hz.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=DEFAULT_CHUNK_SECONDS,
        help="Audio seconds to capture before sending each chunk.",
    )
    parser.add_argument(
        "--read-ms",
        type=int,
        default=DEFAULT_READ_MS,
        help="Microphone read size in milliseconds.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language hint, for example 'en'.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional input device name or index for sounddevice.",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=DEFAULT_QUEUE_SIZE,
        help="Maximum queued audio chunks waiting for transcription.",
    )
    return parser.parse_args()


def load_api_key() -> str:
    load_dotenv(Path(__file__).parent / ".env")
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError("MISTRAL_API_KEY not found. Check your .env file.")
    return api_key


def pcm_to_wav_bytes(
    pcm_bytes: bytes,
    *,
    sample_rate: int,
    channels: int,
    sample_width: int,
) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return buffer.getvalue()


def format_timestamp(seconds: float) -> str:
    minutes, secs = divmod(max(seconds, 0.0), 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    return f"{minutes:02d}:{secs:06.3f}"


def format_speaker(raw_speaker_id: object) -> str:
    if raw_speaker_id is None:
        return "Speaker ?"
    if not isinstance(raw_speaker_id, str):
        return "Speaker ?"
    if not raw_speaker_id.strip():
        return "Speaker ?"
    return f"Speaker {raw_speaker_id}"


def transcribe_chunk(
    *,
    client: httpx.Client,
    api_key: str,
    job: ChunkJob,
    args: argparse.Namespace,
    printer: LivePrinter,
) -> None:
    wav_bytes = pcm_to_wav_bytes(
        job.pcm_bytes,
        sample_rate=args.sample_rate,
        channels=DEFAULT_CHANNELS,
        sample_width=DEFAULT_SAMPLE_WIDTH,
    )
    file_name = f"chunk-{job.index:04d}.wav"
    preview_text = ""

    form_fields: list[tuple[str, str]] = [
        ("model", args.model),
        ("stream", "true"),
        ("diarize", "true"),
    ]
    if args.language:
        form_fields.append(("language", args.language))
    else:
        # The API docs list "segment" and "word" as supported values. For this
        # near-live terminal output, segment timestamps are sufficient.
        form_fields.append(("timestamp_granularities", "segment"))

    files = {
        "file": (file_name, wav_bytes, "audio/wav"),
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/event-stream",
    }

    with client.stream(
        "POST",
        "https://api.mistral.ai/v1/audio/transcriptions",
        headers=headers,
        data=form_fields,
        files=files,
    ) as response:
        if response.status_code != 200:
            raise RuntimeError(
                f"API error occurred: Status {response.status_code}. Body: {response.text}"
            )

        stream = eventstreaming.EventStream(
            response,
            lambda raw: unmarshal_json(raw, TranscriptionStreamEvents),
        )

        for event in stream:
            if not isinstance(event, TranscriptionStreamEvents):
                printer.line(f"[chunk {job.index}] unexpected event: {event}")
                continue

            payload = event.data
            if isinstance(payload, TranscriptionStreamLanguage):
                printer.line(f"[chunk {job.index}] language={payload.audio_language}")
            elif isinstance(payload, TranscriptionStreamSegmentDelta):
                abs_start = job.start_sec + payload.start
                abs_end = job.start_sec + payload.end
                speaker = format_speaker(getattr(payload, "speaker_id", None))
                printer.preview(
                    f"[live {format_timestamp(abs_start)} -> {format_timestamp(abs_end)}] "
                    f"{speaker}: {payload.text}"
                )
            elif isinstance(payload, TranscriptionStreamTextDelta):
                preview_text += payload.text
                if preview_text.strip():
                    printer.preview(f"[live chunk {job.index}] {preview_text}")
            elif isinstance(payload, TranscriptionStreamDone):
                printer.clear_preview()
                if payload.segments:
                    for segment in payload.segments:
                        abs_start = job.start_sec + segment.start
                        abs_end = job.start_sec + segment.end
                        speaker = format_speaker(getattr(segment, "speaker_id", None))
                        printer.line(
                            f"[{format_timestamp(abs_start)} -> {format_timestamp(abs_end)}] "
                            f"{speaker}: {segment.text}"
                        )
                else:
                    text = payload.text.strip()
                    if text:
                        printer.line(
                            f"[{format_timestamp(job.start_sec)} -> "
                            f"{format_timestamp(job.start_sec + args.chunk_seconds)}] "
                            f"Speaker ?: {text}"
                        )
            else:
                printer.line(f"[chunk {job.index}] unhandled event: {event.event}")


def transcription_worker(
    *,
    jobs: "queue.Queue[Optional[ChunkJob]]",
    args: argparse.Namespace,
    api_key: str,
    printer: LivePrinter,
) -> None:
    client = httpx.Client(timeout=60.0)

    while True:
        job = jobs.get()
        if job is None:
            jobs.task_done()
            client.close()
            return

        try:
            transcribe_chunk(
                client=client,
                api_key=api_key,
                job=job,
                args=args,
                printer=printer,
            )
        except Exception as exc:
            printer.line(f"[chunk {job.index}] transcription failed: {exc}")
        finally:
            jobs.task_done()


def capture_audio(
    *,
    args: argparse.Namespace,
    jobs: "queue.Queue[Optional[ChunkJob]]",
    printer: LivePrinter,
) -> None:
    bytes_per_second = args.sample_rate * DEFAULT_CHANNELS * DEFAULT_SAMPLE_WIDTH
    chunk_bytes = int(bytes_per_second * args.chunk_seconds)
    read_frames = max(1, int(args.sample_rate * args.read_ms / 1000))

    current_chunk = bytearray()
    chunk_index = 0
    chunk_start_sec = 0.0

    stream = sd.RawInputStream(
        samplerate=args.sample_rate,
        blocksize=read_frames,
        device=args.device,
        channels=DEFAULT_CHANNELS,
        dtype="int16",
    )

    printer.line(
        "Listening. Press Ctrl+C to stop. "
        f"Submitting {args.chunk_seconds:.1f}s chunks to {args.model}."
    )

    with stream:
        while True:
            data, overflowed = stream.read(read_frames)
            if overflowed:
                printer.line("Warning: microphone input overflowed.")

            current_chunk.extend(bytes(data))

            while len(current_chunk) >= chunk_bytes:
                pcm_bytes = bytes(current_chunk[:chunk_bytes])
                del current_chunk[:chunk_bytes]

                job = ChunkJob(
                    index=chunk_index,
                    start_sec=chunk_start_sec,
                    pcm_bytes=pcm_bytes,
                )
                try:
                    jobs.put_nowait(job)
                except queue.Full:
                    printer.line(
                        f"Dropping chunk {chunk_index}: transcription queue is full."
                    )

                chunk_index += 1
                chunk_start_sec += args.chunk_seconds


def main() -> int:
    args = parse_args()
    api_key = load_api_key()
    printer = LivePrinter()
    jobs: "queue.Queue[Optional[ChunkJob]]" = queue.Queue(maxsize=args.queue_size)

    worker = threading.Thread(
        target=transcription_worker,
        kwargs={
            "jobs": jobs,
            "args": args,
            "api_key": api_key,
            "printer": printer,
        },
        daemon=True,
    )
    worker.start()

    try:
        capture_audio(args=args, jobs=jobs, printer=printer)
    except KeyboardInterrupt:
        printer.line("\nStopping...")
    finally:
        jobs.put(None)
        worker.join()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
