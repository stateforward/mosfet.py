"""The given voice: a canonical self-fact row, written once per source and recalled oldest first."""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pydantic
import pytest

from mosfet.abilities import memory
from mosfet.abilities.vocal import voice

from tests.bot.abilities.support import shared_hsm_context, start_abilities_for_test


class _Damaged(pydantic.BaseModel):
    """A self-fact row under the voice's tag that is not a given voice."""

    value: str


_GIVEN = voice.Given(
    ref=voice.Ref(source="lab", key="v-3f2a9c41d07e"),
    origin="designed",
    description="A bright, quick young voice with a smile in it.",
    languages=("en",),
    traits=("curious",),
)


def test_a_given_voice_holds_canonical_fields_only() -> None:
    schema = voice.Given.model_json_schema()

    assert set(schema["properties"]) == {"ref", "origin", "description", "descriptors", "languages", "traits", "kind"}
    assert _GIVEN.kind == "fact"
    text = json.dumps(schema).lower()
    for provider in ("elevenlabs", "gemini", "openai", "breeze", "kokoro"):
        assert provider not in text
    for invalid in ({"origin": "cloned"}, {"traits": ()}, {"description": ""}, {"kind": "instruction"}):
        with pytest.raises(pydantic.ValidationError):
            _ = voice.Given.model_validate(_GIVEN.model_dump() | invalid)


def test_given_voices_are_self_fact_rows_recalled_oldest_first_skipping_damaged_rows() -> None:
    other = _GIVEN.model_copy(update={"ref": voice.Ref(source="cloud", key="aria")})

    async def run() -> tuple[voice.Given, ...]:
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        try:
            store = memory.Memory(engine=memory.store.open_sqlite_engine(connection=connection))
            ctx = shared_hsm_context()
            await start_abilities_for_test(ctx, store)
            for given in (_GIVEN, other):
                _ = await memory.transact(ctx, store, voice.given_insert_input(given))
            _ = await memory.transact(
                ctx, store, memory.self_fact_insert_input(voice.VOICE_QUERY, _Damaged(value="damaged"))
            )
            return voice.recalled_given(await memory.transact(ctx, store, voice.given_select_input()))
        finally:
            connection.close()

    assert asyncio.run(run()) == (_GIVEN, other)
    assert voice.VOICE_QUERY == "self_voice"
