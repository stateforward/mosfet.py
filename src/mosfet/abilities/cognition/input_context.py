from .. import processing

import typing

TEmbedded: typing.TypeAlias = object


def carry_input_context(
    host: processing.InputData,
    input: TEmbedded,
) -> processing.InputData:
    """Carry schemas and actors into a nested cognitive processor input."""

    return processing.InputData(
        input=input,
        schemas=host.schemas,
        actors=host.actors,
    )


def embedded_input_context(
    host: processing.InputData,
) -> processing.InputData:
    """Return an input copy suitable for embedding inside processor input payloads."""

    return host


__all__ = ["carry_input_context", "embedded_input_context"]
