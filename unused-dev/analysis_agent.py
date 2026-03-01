"""
analysis_agent.py — Analyze text output with a Mistral chat model.

Usage:
  python analysis_agent.py --input transcription.txt
  python analysis_agent.py --input transcription.txt --output analysis.txt
  python analysis_agent.py --input notes.txt --instructions "Focus on action items."
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from mistralai import Mistral


DEFAULT_ANALYSIS_MODEL = "mistral-large-latest"
DEFAULT_INPUT = Path(__file__).parent / "transcription.txt"
DEFAULT_OUTPUT = Path(__file__).parent / "analysis.txt"
DEFAULT_SYSTEM_PROMPT = """You are a Dungeons & Dragons session analysis agent.
Analyze the provided output and respond with these sections:
- Summary
- Important facts
- Entities (characters, locations, items, factions)
- Intent and next actions
- Gaps or ambiguities
- Recommended response

If the text is not about D&D, still analyze it clearly using the same structure."""


@dataclass(frozen=True)
class AnalysisResult:
    text: str
    model: str


def load_api_key() -> str:
    load_dotenv(Path(__file__).parent / ".env")
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError("MISTRAL_API_KEY not found. Check your .env file.")
    return api_key


def build_system_prompt(extra_instructions: str | None = None) -> str:
    if not extra_instructions:
        return DEFAULT_SYSTEM_PROMPT
    return f"{DEFAULT_SYSTEM_PROMPT}\n\nAdditional instructions:\n{extra_instructions.strip()}"


def extract_message_text(content: object) -> str:
    if isinstance(content, str):
        return content

    parts: list[str] = []
    if isinstance(content, list):
        for item in content:
            text = None
            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = getattr(item, "text", None)
            if text:
                parts.append(text)

    merged = "\n".join(part.strip() for part in parts if part.strip()).strip()
    if merged:
        return merged

    raise ValueError("The analysis model returned an unsupported response format.")


def analyze_text(
    text: str,
    *,
    model: str = DEFAULT_ANALYSIS_MODEL,
    extra_instructions: str | None = None,
) -> AnalysisResult:
    if not text.strip():
        raise ValueError("Cannot analyze empty text.")

    client = Mistral(api_key=load_api_key())
    response = client.chat.complete(
        model=model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": build_system_prompt(extra_instructions),
            },
            {
                "role": "user",
                "content": f"Analyze this output:\n\n{text.strip()}",
            },
        ],
    )

    message = response.choices[0].message
    return AnalysisResult(
        text=extract_message_text(message.content).strip(),
        model=model,
    )


def save_analysis(text: str, output_path: Path) -> None:
    output_path.write_text(text, encoding="utf-8")
    print(f"🧠  Analysis saved → {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze an output text file with a Mistral model."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Text file to analyze.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="File where the analysis will be written.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_ANALYSIS_MODEL,
        help="Chat model used for analysis.",
    )
    parser.add_argument(
        "--instructions",
        default=None,
        help="Optional extra analysis instructions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.input.read_text(encoding="utf-8")
    result = analyze_text(
        text,
        model=args.model,
        extra_instructions=args.instructions,
    )

    print("=" * 60)
    print("ANALYSIS:")
    print(result.text)
    print("=" * 60)
    save_analysis(result.text, args.output)


if __name__ == "__main__":
    main()
