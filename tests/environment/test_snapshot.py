"""Direct unit tests for the model-facing snapshot renderer (pure presentation)."""

from bot.environment import snapshot


def test_render_environment_wraps_self_with_world_identity() -> None:
    rendered = snapshot.render_environment("env-1", "<self/>")

    assert rendered == '<environment id="env-1">\n<self/>\n</environment>'


def test_render_environment_quotes_identity() -> None:
    rendered = snapshot.render_environment('a"b', "<self/>")

    assert rendered == "<environment id='a\"b'>\n<self/>\n</environment>"
    assert "<self/>" in rendered


def test_model_repr_marks_values_that_own_their_serialization() -> None:
    class _Compact:
        def __model_repr__(self) -> str:
            return "<compact/>"

    assert isinstance(_Compact(), snapshot.ModelRepr)
    assert not isinstance(object(), snapshot.ModelRepr)
