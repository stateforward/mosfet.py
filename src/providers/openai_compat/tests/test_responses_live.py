"""Live OpenAI Responses API check: reasoning effort high with a required function tool.

Run with an OpenAI key in the environment:
  python -m pytest src/providers/openai_compat/tests/test_responses_live.py -m live -v
"""

from __future__ import annotations

import asyncio
import os

import pytest

import mosfet.providers.openai_compat as openai_compat
from mosfet.abilities.language import text

pytestmark = pytest.mark.live

_DEFAULT_MODEL = "gpt-5.6-luna"


def test_responses_generator_calls_function_tool_with_high_reasoning_effort() -> None:
    api_key = os.environ.get("BOT_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OpenAI API key not set (BOT_OPENAI_API_KEY / OPENAI_API_KEY)")
    generator = openai_compat.ResponsesTextGenerator(
        client=openai_compat.ChatClient(model=os.environ.get("BOT_REASONING_MODEL") or _DEFAULT_MODEL, api_key=api_key),
        provider="openai_live",
        reasoning_effort=openai_compat.ReasoningEffort.HIGH,
    )
    tool = {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two integers.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }

    output = asyncio.run(
        generator.generate(
            text.InputData(
                messages=(text.TextMessage(role=text.TextRole.USER, content="Use the tool to add 2 and 3."),),
                tools=(tool,),
                tool_selection=text.ToolSelectionPolicy.REQUIRED,
            )
        )
    )

    assert [(call.name, call.args) for call in output.tool_calls] == [("add", {"a": 2, "b": 3})]
