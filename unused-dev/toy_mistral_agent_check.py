"""
toy_mistral_agent_check.py — minimal Strands + Mistral smoke test.

Usage:
  python toy_mistral_agent_check.py
  python toy_mistral_agent_check.py --model mistral-large-latest
"""

from __future__ import annotations

import argparse

from pydantic import BaseModel, Field
from strands import Agent, tool
from strands.models.mistral import MistralModel

from analysis_agent import load_api_key


DEFAULT_MODEL = "mistral-small-latest"

EXPECTED_SCENE = {
    "scene_id": "vault-7-blue-moon",
    "party_name": "The Brass Meteors",
    "enemy": "Clockwork Basilisk",
    "mood": "tense",
}


class SceneSmokeCheck(BaseModel):
    status: str = Field(description="Return 'ok' if the tool values were retrieved correctly.")
    scene_id: str = Field(description="The exact scene_id returned by the tool.")
    party_name: str = Field(description="The exact party_name returned by the tool.")
    enemy: str = Field(description="The exact enemy returned by the tool.")
    mood: str = Field(description="The exact mood returned by the tool.")


@tool
def get_test_scene() -> dict[str, str]:
    """Return a fixed fake D&D scene for smoke testing the Mistral Strands agent."""
    return dict(EXPECTED_SCENE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal Strands + Mistral smoke test."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Mistral model ID to use for the smoke test.",
    )
    return parser.parse_args()


def build_agent(model_id: str) -> Agent:
    model = MistralModel(
        api_key=load_api_key(),
        model_id=model_id,
        temperature=0.1,
        max_tokens=220,
        stream=False,
    )
    return Agent(
        model=model,
        name="Toy Mistral Smoke Test",
        description="Checks that a Strands agent can answer and use a tool via Mistral.",
        system_prompt=(
            "You are a strict smoke-test agent. "
            "Follow the user's instructions exactly and keep responses short."
        ),
        tools=[get_test_scene],
    )


def validate_scene(result: SceneSmokeCheck) -> list[str]:
    errors: list[str] = []
    if result.status.strip().lower() != "ok":
        errors.append(f"status={result.status!r}")

    for key, expected_value in EXPECTED_SCENE.items():
        actual_value = getattr(result, key)
        if actual_value != expected_value:
            errors.append(f"{key}={actual_value!r} expected {expected_value!r}")

    return errors


def main() -> None:
    args = parse_args()
    agent = build_agent(args.model)

    basic_result = agent("Reply with exactly: mistral agent ok")
    basic_text = str(basic_result).strip()

    structured_result = agent(
        (
            "Use the get_test_scene tool. Copy the exact values from the tool into the "
            "structured output. Set status to ok only if you retrieved the values from the tool."
        ),
        structured_output_model=SceneSmokeCheck,
    )

    if not structured_result.structured_output:
        raise SystemExit("Smoke test failed: no structured output returned.")

    scene_check = structured_result.structured_output
    errors = validate_scene(scene_check)

    print("Basic response:")
    print(basic_text)
    print()
    print("Structured tool check:")
    print(scene_check.model_dump_json(indent=2))

    if errors:
        raise SystemExit(
            "Smoke test failed:\n- " + "\n- ".join(errors)
        )

    print()
    print(f"Smoke test passed with model {args.model}.")


if __name__ == "__main__":
    main()
