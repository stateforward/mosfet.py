from bot.abilities import memory


def test_memory_payload_contract_round_trips_binary_parts_and_vectors() -> None:
    input = memory.payload.EncodeData(
        memory_id="mem-operator-prefers-terse-notes",
        kind=memory.classification.MemoryClassificationKind.PREFERENCE,
        sensitivity=memory.classification.MemoryClassificationSensitivity.STANDARD,
        participant_ref="operator",
        metadata={"source": "conversation", "turn": 7},
        parts=(
            memory.payload.MemoryPart(
                modality=memory.payload.MemoryModality.TEXT,
                content_type="text/plain; charset=utf-8",
                data=b"The operator prefers terse handoff notes.",
                metadata={"lang": "en"},
            ),
            memory.payload.MemoryPart(
                modality=memory.payload.MemoryModality.IMAGE,
                content_type="image/png",
                data=b"\x89PNG\r\n\x1a\n",
            ),
        ),
        vectors=(
            memory.payload.MemoryVector(
                embedding=(0.1, 0.2, 0.3),
                model="test-embedding-3d",
                part_index=0,
                metadata={"distance": "cosine"},
            ),
        ),
    )

    encoded = input.model_dump_json()
    decoded = memory.payload.EncodeData.model_validate_json(encoded)
    output = memory.payload.DecodedData(
        memory_id="mem-operator-prefers-terse-notes",
        kind=decoded.kind,
        sensitivity=decoded.sensitivity,
        participant_ref=decoded.participant_ref,
        metadata=decoded.metadata,
        parts=decoded.parts,
        vectors=decoded.vectors,
    )

    assert "iVBORw0KGgo=" in encoded
    assert decoded == input
    assert output.parts == input.parts
    assert output.vectors == input.vectors


def test_memory_decode_input_is_the_provider_neutral_lookup_contract() -> None:
    input = memory.payload.DecodeData(memory_id="mem-operator-prefers-terse-notes")

    assert input.memory_id == "mem-operator-prefers-terse-notes"
    assert memory.payload.DecodeData.model_json_schema()["properties"]["memory_id"]["minLength"] == 1
